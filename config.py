from dataclasses import dataclass


root_path = "./data"
out_path = "./data"


@dataclass
class Fields:
    cfips: str = "cfips"
    row_id: str = "row_id"
    date: str = "first_day_of_month" 
    county: str = "county"
    state: str = "state"
    month: str = "month"
    md: str = "microbusiness_density"
    active: str = "active"
    pop: str = "population"
