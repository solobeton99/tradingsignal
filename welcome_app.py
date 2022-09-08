import streamlit as st
import numpy as np
import pandas as pd
from hydralit import HydraHeadApp
import matplotlib as plt
from datetime import date
today = date.today()
from datetime import date, datetime
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt
from datetime import date, datetime

from cryptocmd import CmcScraper
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

from cryptocmd import CmcScraper
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


class WelcomeApp(HydraHeadApp):

#wrap all your code in this method and you should be done
    def run(self):
        st.write("this is the best place to be")

