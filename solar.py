import math
import matplotlib.pyplot as plt
import numpy as np

d_0 = 4 # date of perihelion
S_0 = 1366 # flux at TOA at perihelion
P = 365.24 # period of earth's rotation
D_0 = 1.47e8 # distance of Earth from sun at perihelion (m)
e = .0167 # eccentricity of earth's orbit

d_r = 172 # date of summer solstice (173 for leap, 172 for normal)
d_y = 365 # days in the year (366 for leap)
phi_r = 23.44 # tilt of earth on axis relative to perindicular line to the elliptic


def cos_d(x):
    """
    Returns the cosine of x, assuming x is in degrees
    """
    return math.cos(x * math.pi / 180)

def sin_d(x):
    """
    Returns the sine of x, assuming x is in degrees
    """
    return math.sin(x * math.pi / 180)

def sun_to_earth_distance(d, d_0, P, D_0, e):
    """
    Inputs:
        d_0: the perihelion of Earth's orbit (January 4)
        P: period of rotation around the sun (days)
        D_0: distance of Earth from sun at perihelion (m)
        e: eccentricity of orbit
    """
    M = 2 * math.pi * (d - d_0) / P
    D = D_0 * (1- e ** 2) / (1 + e * math.cos(M)) # Distance from sun at day d
    return D

def get_S_toa(S_0, D_0, D):
    """
    Inputs:
        S_0: approximate solar flux (W / m^2)
        D_0: distance of Earth from sun at perihelion (m)
        D: the distance of Earth from sun on the current day 
    """
    return S_0 * ( (D_0 / D) ** 2 ) # Solar flux on earth at day d

def get_solar_declination_angle(d, d_r, d_y, phi_r):
    """
    Inputs: 
        d: relative julian day
        d_r: date of summer solstice (173 for leap, 172 for normal)
        d_y: days in the year (366 for leap)
        phi_r: tilt of earth on axis relative to perpindicular line to the elliptic

    Returns:
        Solar declination angle (angle between solar rays, earth's equator). Negative implies that elliptic plane is below equator (so summer, for people in the north)
    """

    delta_s = phi_r * math.cos(2 * math.pi * (d - d_r) / d_y) # 
    return delta_s

def get_solar_elevation_angle(longitude, latitude, t_d, delta_s, utc_diff=-5):
    """
    longitude: positive east, negative west
    latitude: positive north, negative south
    t_d: hr of day at location
    delta_s: solar declination angle (degrees)
    
    utc_diff: time difference between current location, UTC (Greensboro, UK). For Ithaca, this is -5
    """
    t_utc = t_d - utc_diff


    #sin_psi = sin_d(latitude) * sin_d(delta_s) - cos_d(latitude)*cos_d(delta_s)*cos_d(longitude + 360 * t_utc / t_d) #*cos_d(15 * (t_d - 12))#
    sin_psi = sin_d(latitude) * sin_d(delta_s) - cos_d(latitude)*cos_d(delta_s)*cos_d(15*(t_d - 24))

    psi = 180 / math.pi * math.asin(sin_psi) # Local elevation angle of sun above horizon, at a given time and location

    zenith = 360 / 4 - (180 / math.pi)*psi # Local zenith angle (complement of elevation angle)
    return psi

distances = []
def get_S_TOAs():
    s_TOAs = []
    for day in range(1, 366):
        d = sun_to_earth_distance(day, d_0, P, D_0, e)
        distances.append(d)
        s_TOAs.append(get_S_toa(S_0, D_0, d))
    return s_TOAs


def get_F_transmitted(F_toa, solar_elevation_angle, mean_optical_depth, tropo_height=14e3):
    # mean_optical_depth is optical depth at elevation_angle = 90
    # takes into account pathlength, but assumes that the mean optical depth is the same as the optical depth at 90 degrees
    # not certain that this is true, although it might be (symettric orbit)
    if solar_elevation_angle <= 0:
        return 0
    optical_depth = mean_optical_depth / sin_d(solar_elevation_angle)  # could also just use the mean for this

    F_transmitted = F_toa * math.exp(-optical_depth)
    return F_transmitted


def get_F_panel(solar_elevation_angle, panel_elevation_angle, F_90):
    """
    solar elevation angle: angle of elevation of sun
    panel elevation angle: angle of elevation of panels
    F_90: flux at 90 degrees (after atmospheric attenuation)
    """
    cos_incident = cos_d(solar_elevation_angle)*sin_d(panel_elevation_angle) + sin_d(solar_elevation_angle)*cos_d(panel_elevation_angle)
    incidence_angle = 180 / math.pi * math.acos(cos_incident)
    F_panel = cos_incident * F_90

    return incidence_angle, F_panel



def get_flux_per_day(S_TOAs,latitude = 42.8680, longitude = 76.9856, mean_optical_depth = .2):
    """
    Assumes cloudless (biased high)
    Assumes same pathlength (not correcting for aerosol optical depth, so will still be high)
    """
    year = []
    dec_angles = []
    elevation_angles = []
    for day in range(1, 366): # assume leap year
        delta_s = get_solar_declination_angle(day, d_r, d_y, phi_r)
        dec_angles.append(delta_s)

        day_data = []
        F_TOA = S_TOAs[day - 1]
        for hour in range(1, 25):
            elevation_angle = get_solar_elevation_angle(longitude, latitude, hour, delta_s)
            elevation_angles.append(elevation_angle)
            F_90 = get_F_transmitted(F_TOA, elevation_angle, mean_optical_depth)
            
            
            angle, F_panel = get_F_panel(elevation_angle, latitude, F_90)
            
            day_data.append(F_panel)
        
        year.append(day_data)
    return year

def get_radiation_prior(latitude = 42.8680, longitude = 76.9856):
    s_TOAs = get_S_TOAs()
    all_energy_data = get_flux_per_day(s_TOAs, latitude, longitude)
    return np.array(all_energy_data).flatten()

def get_daily_sums(all_energy_data):
    return [sum(day) / 1000 for day in all_energy_data]



if __name__ == "__main__":
    rad = get_radiation_prior()
   # plt.imshow(rad.T, cmap='hot', interpolation='nearest')
    plt.plot(rad)
    plt.title("Solar flux at Ground in Geneva, NY per Hour")
    plt.xlabel('Hour of year')
    plt.ylabel('Flux')
    plt.show()