import numpy as np

RAUTH = 6371007.2 # Earth authalic radius (m)


def authalic_latitude(phi, ecc):
    qp   =   1   -   ( (1 - ecc**2) /(2*ecc) ) * np.log( (1 - ecc) / (1 + ecc) )
    q    =     (1 - ecc**2)*np.sin(phi) / (1 - (ecc*np.sin(phi))**2)   -   ( (1 - ecc**2) /(2*ecc) ) * np.log( (1 - ecc*np.sin(phi)) / (1 + ecc*np.sin(phi)) )
    return np.arcsin( q / qp )


def conformal_latitude(phi, ecc):
    return 2*np.sqrt(np.arctan(((1 + np.sin(phi))/(1 - np.sin(phi)))*(((1 - ecc*np.sin(phi))/(1 + ecc*np.sin(phi)))**ecc))) - np.pi/2


def authalic_radius(a, ecc):
    return a * np.sqrt( 0.5*(    1    +    ((1-ecc**2) / ecc)  *  np.log( (1+ecc) / np.sqrt(1-ecc**2) )   ))


def great_circle_distance(lon0, lat0, lon1, lat1):
    """
    This function computes the distance between 2 points A and B on Earth with spherical approximation and
    using authalic radius (arguments: lonA, lat_a, lonN, latB)
    """
    return RAUTH * np.arccos( np.sin(np.deg2rad(lat0))*np.sin(np.deg2rad(lat1)) + np.cos(np.deg2rad(lat0))*np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lon1-lon0)) ) 


def cell_area_sphere(x0, y0, x1, y1, r):
    '''
    Compute the area of a longitude-latitude rectangle "x0,y0,x1,y1" on a sphere of radius "r" x0, y0, x1, y1 must be in RADIANS
    x0, y0, x1, y1 must be numpy array of shape: (nx,) (nx,) (ny,) and (ny,)
    '''
    nx = x0.size
    ny = y0.size
    return (r**2)  *  np.matmul( np.reshape((x1-x0), (nx,1)),
                                 np.reshape((np.sin(y1) - np.sin(y0)), (1,ny)))


def cell_area(lon0, lat0, lon1, lat1, a, ecc=0):
    '''
    Compute the area of a longitude-latitude rectangle 'lon0,lat0,lon1,lat1' in an ellipsoid of great radius 'a' and
    eccentricity 'ecc' (optional, default value is 0 => sphere)
    lon0, lat0, lon1 and lat1 must be in DEGREES
    '''

    lon0 = np.deg2rad(np.array(lon0))
    lat0 = np.deg2rad(np.array(lat0))
    lon1 = np.deg2rad(np.array(lon1))
    lat1 = np.deg2rad(np.array(lat1))

    if lon0.shape == ():
        lon0 = lon0.reshape((1))

    if lat0.shape == ():
        lat0 = lat0.reshape((1))

    if lon1.shape == ():
        lon1 = lon1.reshape((1))

    if lat1.shape == ():
        lat1 = lat1.reshape((1))

    if (ecc>0):
        lat0 = authalic_latitude(lat0, ecc)
        lat1 = authalic_latitude(lat1, ecc)
        a = authalic_radius(a, ecc)

    return cell_area_sphere(lon0, lat0, lon1, lat1, a)

