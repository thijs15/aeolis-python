import logging
import numpy as np

# package modules
from utils import *


# initialize logger
logger = logging.getLogger(__name__)


def initialize(s, p):
    '''Initialize bathymetry and bed composition

    Initialized bathymetry, computes cell sizes and orientation, bed
    layer thickness and bed composition.

    Parameters
    ----------
    s : dict
        Spatial grids
    p : dict
        Model configuration parameters

    Returns
    -------
    dict
        Spatial grids

    '''
    
    # get model dimensions
    nx = p['nx']
    ny = p['ny']
    nl = p['nlayers']
    nf = p['nfractions']

    # initialize x-dimension
    s['x'][:,:] = p['xgrid_file']
    s['ds'][:,1:] = np.diff(s['x'], axis=1)
    s['ds'][:,0] = s['ds'][:,1]

    # initialize y-dimension
    if ny == 0:
        s['y'][:,:] = 0.
        s['dn'][:,:] = 1.
        s['alfa'][:,:] = 0.
    else:
        s['y'][:,:] = p['ygrid_file']
        s['dn'][1:,:] = np.diff(s['y'], axis=0)
        s['dn'][0,:] = s['dn'][1,:]

        s['alfa'][1:-1,:] = np.arctan2(s['x'][2:,:] - s['x'][:-2,:],
                                       s['y'][2:,:] - s['y'][:-2,:])
        s['alfa'][0,:] = s['alfa'][1,:]
        s['alfa'][-1,:] = s['alfa'][-2,:]

    # compute cell areas
    s['dsdn'][:,:] = s['ds'] * s['dn']
    s['dsdni'][:,:] = 1. / s['dsdn']

    # initialize bathymetry
    s['zb'][:,:] = p['bed_file']

    # initialize bed layers
    s['thlyr'][:,:,:] = p['layer_thickness']

    # initialize bed composition
    if p['bedcomp_file'] is None:
        gs = makeiterable(p['grain_dist'])
        gs = gs / np.sum(gs)
        for i in range(nl):
            for j in range(nf):
                s['mass'][:,:,i,j] = p['rhop'] * p['porosity'] \
                                     * s['thlyr'][:,:,i] * gs[j]
    else:
        s['mass'][:,:,:,:] = p['bedcomp_file'].reshape(s['mass'].shape)

    # initialize active layer
    l = np.log(1.-.99) / p['dza']
    dzp = np.exp(l * p['layer_thickness'] * np.arange(nl))
    s['dzp'] = dzp.reshape((1,1,-1)).repeat(ny+1, axis=0).repeat(nx+1, axis=1)

    return s


def update(s, p):
    '''Update bathymetry and bed composition

    Update bed composition by moving sediment fractions between bed
    layers. The total mass in a single bed layer does not change as
    sediment removed from a layer is repleted with sediment from
    underlying layers. Similarly, excess sediment added in a layer is
    moved to underlying layers in order to keep the layer mass
    constant. The lowest bed layer exchanges sediment with an infinite
    sediment source that follows the original grain size distribution
    as defined in the model configuration file by ``grain_size`` and
    ``grain_dist``. The bathymetry is updated following the
    cummulative erosion/deposition over the fractions if ``bedupdate``
    is ``True``.

    Parameters
    ----------
    s : dict
        Spatial grids
    p : dict
        Model configuration parameters

    Returns
    -------
    dict
        Spatial grids

    '''

    nx = p['nx']
    ny = p['ny']
    nl = p['nlayers']
    nf = p['nfractions']

    # reshape mass matrices
    pickup = s['pickup'].reshape((-1,nf))
    m = s['mass'].reshape((-1,nl,nf))
    dzp = s['dzp'].reshape((-1,nl,1)).repeat(nf, axis=-1)

    # update bed based on pickup
    m0 = m.sum(axis=-1, keepdims=True)
    for i in range(nl):
        mp = m[:,i,:] * dzp[:,i,:]
        ix = mp != 0.
        mf = np.zeros(mp.shape)
        mf[ix] = np.minimum(1., pickup[ix] / mp[ix])
        
        pickup -= mp * mf
        m[:,i,:] -= mp * mf

    # move mass among layers
    for i1 in range(nl):
        for i2 in range(i1+1,nl):

            d1 = normalize(m[:,i1,:], axis=1)
            d2 = normalize(m[:,i2,:], axis=1)

            dm = m[:,i1,:].sum(axis=-1, keepdims=True) - m0[:,i1,:]
            if np.all(dm == 0.):
                break
            
            ix_ero = (dm < 0.).flatten()
            ix_dep = (dm > 0.).flatten()

            dm[ix_ero,:] = np.minimum(m[ix_ero,i2,:].sum(axis=-1, keepdims=True), dm[ix_ero,:])
            dmr = dm.repeat(nf, axis=-1)
            
            m[ix_ero,i1,:] -= dmr[ix_ero,:] * d2[ix_ero,:]
            m[ix_ero,i2,:] += dmr[ix_ero,:] * d2[ix_ero,:]
            m[ix_dep,i1,:] -= dmr[ix_dep,:] * d1[ix_dep,:]
            m[ix_dep,i2,:] += dmr[ix_dep,:] * d1[ix_dep,:]

    # remove/add mass from/to base layer
    d = normalize(m[:,-1,:], axis=1)
    dm = m[:,-1,:].sum(axis=-1, keepdims=True) - m0[:,-1,:]
    ix_ero = (dm < 0.).flatten()
    ix_dep = (dm > 0.).flatten()
    dmr = dm.repeat(nf, axis=-1)
    
    m[ix_dep,-1,:] -= dmr[ix_dep,:] * d[ix_dep,:]
    m[ix_ero,-1,:] -= dmr[ix_ero,:] * normalize(p['grain_dist'])[np.newaxis,:].repeat(np.sum(ix_ero), axis=0)
        
    # remove tiny negatives
    m = prevent_tiny_negatives(m, p['max_error'])

    # warn if not all negatives are gone
    if m.min() < 0:
        logger.warn(format_log('Negative mass',
                               nrcells=np.sum(np.any(m<0., axis=-1)),
                               minvalue=m.min(),
                               minwind=s['uw'].min(),
                               time=p['_time']))
        
    # reshape mass matrix
    s['mass'] = m.reshape((ny+1,nx+1,nl,nf))

    # update bathy
    if p['bedupdate']:
        s['zb'] += dm[:,0].reshape((ny+1,nx+1)) / (p['rhop'] * p['porosity'])

    return s


def prevent_negative_mass(m, dm, pickup):
    '''Handle situations in which negative mass may occur due to numerics

    Negative mass may occur by moving sediment to lower layers down to
    accomodate deposition of sediments. In particular two cases are
    important:

    #. A net deposition cell has some erosional fractions.

       In this case the top layer mass is reduced according to the
       existing sediment distribution in the layer to accomodate
       deposition of fresh sediment. If the erosional fraction is
       subtracted afterwards, negative values may occur. Therefore the
       erosional fractions are subtracted from the top layer
       beforehand in this function. An equal mass of deposition
       fractions is added to the top layer in order to keep the total
       layer mass constant. Subsequently, the distribution of the
       sediment to be moved to lower layers is determined and the
       remaining deposits are accomodated.

    #. Deposition is larger than the total mass in a layer.

       In this case a non-uniform distribution in the bed may also
       lead to negative values as the abundant fractions are reduced
       disproportionally as sediment is moved to lower layers to
       accomodate the deposits. This function fills the top layers
       entirely with fresh deposits and moves the existing sediment
       down such that the remaining deposits have a total mass less
       than the total bed layer mass. Only the remaining deposits are
       fed to the routine that moves sediment through the layers.

    Parameters
    ----------
    m : np.ndarray
        Sediment mass in bed (nx*ny, nl, nf)
    dm : np.ndarray
        Total sediment mass exchanged between layers (nx*ny, nf)
    pickup : np.ndarray
        Sediment pickup (nx*ny, nf)

    Returns
    -------
    np.ndarray
        Sediment mass in bed (nx*ny, nl, nf)
    np.ndarray
        Total sediment mass exchanged between layers (nx*ny, nf)
    np.ndarray
        Sediment pickup (nx*ny, nf)

    Note
    ----
    The situations handled in this function can also be prevented by
    reducing the time step, increasing the layer mass or increasing
    the adaptation time scale.

    '''

    nl = m.shape[1]
    nf = m.shape[2]

    ###
    ### case #1: deposition cells with some erosional fractions
    ###
    
    ix_dep = dm[:,0] > 0.
    
    # determine erosion and deposition fractions per cell
    ero =  np.maximum(0., pickup)
    dep = -np.minimum(0., pickup)

    # determine gross erosion
    erog = np.sum(ero, axis=1, keepdims=True).repeat(nf, axis=1)

    # determine net deposition cells with some erosional fractions
    ix = ix_dep & (erog[:,0] > 0)

    # remove erosional fractions from pickup and remove an equal mass
    # of accretive fractions from the pickup, adapt sediment exchange
    # mass and bed composition accordingly
    if np.any(ix):
        d = normalize(dep, axis=1)
        ddep = erog[ix,:] * d[ix,:]
        pickup[ix,:] = -dep[ix,:] + ddep
        dm[ix,:] = -np.sum(pickup[ix,:], axis=-1, keepdims=True).repeat(nf, axis=-1)
        m[ix,0,:] -= ero[ix,:] - ddep # FIXME: do not use deposition in normalization

    ###
    ### case #2: deposition cells with deposition larger than the mass present in the top layer
    ###

    mx = m[:,0,:].sum(axis=-1, keepdims=True)

    # determine deposition in terms of layer mass (round down)
    n = dm[:,:1] // mx

    # determine if deposition is larger than a sinle layer mass
    if np.any(n > 0):

        # determine distribution of deposition
        d = normalize(pickup, axis=1)

        # walk through layers from top to bottom
        for i in range(nl):

            ix = (n > i).flatten()
            if not np.any(ix):
                break

            # move all sediment below current layer down one layer
            m[ix,(i+1):,:] = m[ix,i:-1,:]

            # fill current layer with deposited sediment
            m[ix,i,:] = mx[ix,:].repeat(nf, axis=1) * d[ix,:]

            # remove deposited sediment from pickup
            pickup[ix,:] -= m[ix,i,:]

        # discard any remaining deposits at locations where all layers
        # are filled with fresh deposits
        ix = (dm[:,:1] > mx).flatten()
        if np.any(ix):
            pickup[ix,:] = 0.

        # recompute sediment exchange mass
        dm[ix,:] = -np.sum(pickup[ix,:], axis=-1, keepdims=True).repeat(nf, axis=-1)

    return m, dm, pickup


def mixtoplayer(s, p):
    '''Mix grain size distribution in top layer of the bed

    Simulates mixing of the top layers of the bed by wave action. The
    wave action is represented by a local wave height maximized by a
    maximum wave hieght over depth ratio ``gamma``. The mixing depth
    is a fraction of the local wave height indicated by
    ``facDOD``. The mixing depth is used to compute the number of bed
    layers that should be included in the mixing. The grain size
    distribution in these layers is then replaced by the average grain
    size distribution over these layers.

    Parameters
    ----------
    s : dict
        Spatial grids
    p : dict
        Model configuration parameters

    Returns
    -------
    dict
        Spatial grids

    '''

    if p['mixtoplayer']:
        
        # get model dimensions
        nl = p['nlayers']
        nf = p['nfractions']

        # compute depth of disturbence for each cell and repeat for each layer
        DOD = p['facDOD'] * s['Hs']
        DOD = DOD[:,:,np.newaxis].repeat(nl, axis=2)
        
        # determine what layers are above the depth of disturbance
        ix = (s['thlyr'].cumsum(axis=2) <= DOD) & (DOD > 0.)
        ix = ix[:,:,:,np.newaxis].repeat(nf, axis=3)
        
        # average mass over layers
        if np.any(ix):
            ix[:,:,0,:] = True # at least mix the top layer
            mass = s['mass'].copy()
            mass[~ix] = np.nan
            
            s['mass'][ix] = np.nanmean(mass, axis=2)[:,:,np.newaxis,:].repeat(nl, axis=2)[ix]
        
    return s
