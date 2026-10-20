"""Weather points for PacifiCorp East (Rocky Mountain Power) load centers, with approximate weights by share of
PACE retail load (Wasatch Front dominates; Wyoming load is largely industrial). Weights are an informed approximation
from population and service-territory size, not utility-published figures."""
CITIES = {
    "slc":    dict(name="Salt Lake City UT", lat=40.76, lon=-111.89, w=0.38),
    "ogden":  dict(name="Ogden UT",          lat=41.22, lon=-111.97, w=0.14),
    "provo":  dict(name="Provo UT",          lat=40.23, lon=-111.66, w=0.14),
    "logan":  dict(name="Logan UT",          lat=41.74, lon=-111.83, w=0.05),
    "cedar":  dict(name="Cedar City UT",     lat=37.68, lon=-113.06, w=0.04),
    "rexburg":dict(name="Rexburg ID",        lat=43.83, lon=-111.79, w=0.07),
    "casper": dict(name="Casper WY",         lat=42.87, lon=-106.31, w=0.10),
    "rocksp": dict(name="Rock Springs WY",   lat=41.59, lon=-109.20, w=0.08),
}
