'''This file is part of AeoLiS.
   
AeoLiS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
   
AeoLiS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
   
You should have received a copy of the GNU General Public License
along with AeoLiS.  If not, see <http://www.gnu.org/licenses/>.
   
AeoLiS  Copyright (C) 2015 Bas Hoonhout

bas.hoonhout@deltares.nl         b.m.hoonhout@tudelft.nl
Deltares                         Delft University of Technology
Unit of Hydraulic Engineering    Faculty of Civil Engineering and Geosciences
Boussinesqweg 1                  Stevinweg 1
2629 HVDelft                     2628CN Delft
The Netherlands                  The Netherlands

'''


from __future__ import absolute_import, division

import logging
import numpy as np

# package modules
from aeolis.utils import *


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
                s['mass'][:,:,i,j] = p['rhop'] * (1. - p['porosity']) \
                                     * s['thlyr'][:,:,i] * gs[j]
    else:
        s['mass'][:,:,:,:] = p['bedcomp_file'].reshape(s['mass'].shape)                

    # initialize masks
    for k, v in p.items():
        if k.endswith('_mask'):
            if v is None:
                s[k] = 1.
            else:
                s[k] = v.reshape(s['zb'].shape)

    # initialize threshold
    if p['threshold_file'] is not None:
        s['uth'] = p['threshold_file'][:,:,np.newaxis].repeat(nf, axis=-1)
        
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

    # determine net erosion
    pickup = s['pickup'].reshape((-1,nf))

    # determine total mass that should be exchanged between layers
    dm = -np.sum(pickup, axis=-1, keepdims=True).repeat(nf, axis=-1)
    
    # get erosion and deposition cells
    ix_ero = dm[:,0] < 0.
    ix_dep = dm[:,0] > 0.
    
    # reshape mass matrix
    m = s['mass'].reshape((-1,nl,nf))

    # negative mass may occur in case of deposition due to numerics,
    # which should be prevented
    m, dm, pickup = prevent_negative_mass(m, dm, pickup)
    
    # determine weighing factors
    d = normalize(m, axis=2)
    
    # move mass among layers
    m[:,0,:] -= pickup
    for i in range(1,nl):
        m[ix_ero,i-1,:] -= dm[ix_ero,:] * d[ix_ero,i,:]
        m[ix_ero,i,  :] += dm[ix_ero,:] * d[ix_ero,i,:]
        m[ix_dep,i-1,:] -= dm[ix_dep,:] * d[ix_dep,i-1,:]
        m[ix_dep,i,  :] += dm[ix_dep,:] * d[ix_dep,i-1,:]
    m[ix_dep,-1,:] -= dm[ix_dep,:] * d[ix_dep,-1,:]
    m[ix_ero,-1,:] -= dm[ix_ero,:] * normalize(p['grain_dist'])[np.newaxis,:].repeat(np.sum(ix_ero), axis=0)

    # remove tiny negatives
    m = prevent_tiny_negatives(m, p['max_error'])

    # warn if not all negatives are gone
    if m.min() < 0:
        logger.warning(format_log('Negative mass',
                                  nrcells=np.sum(np.any(m<0., axis=-1)),
                                  minvalue=m.min(),
                                  minwind=s['uw'].min(),
                                  time=p['_time']))
        
    # reshape mass matrix
    s['mass'] = m.reshape((ny+1,nx+1,nl,nf))

    # update bathy
    if p['process_bedupdate']:
        dz = dm[:,0].reshape((ny+1,nx+1)) / (p['rhop'] * (1. - p['porosity']))
        s['zb'] += dz
        s['zs'] += dz

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

    if p['process_mixtoplayer']:
        
        # get model dimensions
        nx = p['nx']+1
        ny = p['ny']+1
        nl = p['nlayers']
        nf = p['nfractions']

        # compute depth of disturbence for each cell and repeat for each layer
        DOD = p['facDOD'] * s['Hs']

        # compute ratio total layer thickness and depth of disturbance 
        ix = DOD > 0.
        f = np.ones(DOD.shape)
        f[ix] = np.minimum(1., s['thlyr'].sum(axis=2)[ix] / DOD[ix])

        # correct shapes
        DOD = DOD[:,:,np.newaxis].repeat(nl, axis=2)
        f = f[:,:,np.newaxis].repeat(nl, axis=2)

        # determine what layers are above the depth of disturbance
        ix = (s['thlyr'].cumsum(axis=2) <= DOD) & (DOD > 0.)
        ix = ix[:,:,:,np.newaxis].repeat(nf, axis=3)
        f = f[:,:,:,np.newaxis].repeat(nf, axis=3)
        
        # average mass over layers
        if np.any(ix):
            ix[:,:,0,:] = True # at least mix the top layer
            mass = s['mass'].copy()
            mass[~ix] = np.nan
            
            gd = normalize(p['grain_dist']) * p['rhop'] * (1. - p['porosity'])
            gd = gd.reshape((1,1,1,-1)).repeat(ny, axis=0) \
                                       .repeat(nx, axis=1) \
                                       .repeat(nl, axis=2)

            mass1 = np.nanmean(mass, axis=2, keepdims=True).repeat(nl, axis=2)
            mass2 = gd * s['thlyr'][:,:,:,np.newaxis].repeat(nf, axis=-1)
            mass = mass1 * f + mass2 * (1. - f)
        
            s['mass'][ix] = mass[ix]
            
    return s


def avalanche(s, p):

    return s
