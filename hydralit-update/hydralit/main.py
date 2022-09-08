from typing import Dict
from tradingview_ta import TA_Handler, Interval
import streamlit as st
from streamlit_autorefresh import st_autorefresh
# after it's been refreshed 100 times.

import streamlit as st
from cryptocmd import CmcScraper
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

import plotly.graph_objects as go
from Functions import *
import pandas_ta as ta
from PIL import Image
import performanceanalytics.statistics as pas
import performanceanalytics.table.table as pat
import investpy as inv

import os
import shutil
import time
from datetime import date
import tradingview_ta, requests, os
from datetime import timezone
from datetime import date
import yfinance as yf
from yahoo_fin.stock_info import get_data
import yahoo_fin.stock_info as si
from yahoo_fin import news
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime as dt

today = date.today()
from st_aggrid import AgGrid
import pandas as pd
import yfinance as yf
from streamlit_echarts import st_echarts
import matplotlib.pyplot as plt
import streamlit as st
import apps
from datetime import datetime, timedelta, timezone
from hydralit.sessionstate import SessionState
from hydralit.loading_app import LoadingApp
import hydralit_components as hc
from hydralit.wrapper_class import Templateapp

import yfinance as yahooFinance
import urllib.request
import json

from hydralit import HydraApp

scripts = import_scripts()

app = HydraApp(
    title='Secure Hydralit Data Explorer',
    favicon="🐙"
   )


class HydraApp(object):
    """
 Class to create a host application for combining multiple streamlit applications.
 """

    def __init__(self,
                 title='Hydralit Apps',
                 nav_container=None,
                 nav_horizontal=True,
                 layout='wide',
                 favicon="🧊",
                 use_navbar=True,
                 navbar_theme=None,
                 navbar_sticky=True,
                 navbar_mode='pinned',
                 use_loader=True,
                 use_cookie_cache=True,
                 sidebar_state='auto',
                 navbar_animation=True,
                 allow_url_nav=False,
                 hide_streamlit_markers=False,
                 use_banner_images=None,
                 banner_spacing=None,
                 clear_cross_app_sessions=True,
                 session_params=None):
        """
  A class to create an Multi-app Streamlit application. This class will be the host application for multiple applications that are added after instancing.
  The secret saurce to making the different apps work together comes from the use of a global session store that is shared with any HydraHeadApp that is added to the parent HydraApp.
  The session store is created when this class is instanced, by default the store contains the following variables that are used across the child apps:
   - previous_app
   = selected_app
   - preserve_state
   - allow_access
   - current_user
  More global values can be added by passing in a Dict when instancing the class, the dict needs to provide a name and default value that will be added to the global session store.
  Parameters
  -----------
  title: str, 'Streamlit MultiApp'
      The title of the parent app, this name will be used as the application (web tab) name.
  nav_container: Streamlit.container, None
      A container in which to populate the navigation buttons for each attached HydraHeadApp. Default will be a horizontal aligned banner style over the child applications. If the Streamlit sidebar is the target container, the navigation items will appear at the top and the default state of the sidebar will be expanded.
  nav_horizontal: bool, True
      To align the navigation buttons horizonally within the navigation container, if False, the items will be aligned vertically.
  layout: str, 'wide'
      The layout format to be used for all app pages (HydraHeadApps), same as the layout variable used in `set_page_config <https://docs.streamlit.io/en/stable/api.html?highlight=set_page_config#streamlit.set_page_config>`.
  favicon: str
      An inline favicon image to be used as the application favicon.
  allow_url_nav: bool False
      Enable navigation using url parameters, this allows for bookmarking and using internal links for navigation
  use_navbar: bool, False
      Use the Hydralit Navbar component or internal Streamlit components to create the nav menu. Currently Hydralit Navbar doesn't support dropdown menus.
  navbar_theme: Dict, None
      Override the Hydralit Navbar theme, you can overrider only the part you wish or the entire theme by only providing details of the changes.
       - txc_inactive: Inactive Menu Item Text color
       - menu_background: Menu Background Color
       - txc_active: Active Menu Item Text Color
       - option_active: Active Menu Item Color
      example, navbar_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
  navbar_sticky: bool, True
      Set navbar to be sticky and fixed to the top of the window.
  use_loader: bool, True
      Set if to use the app loader with auto transition spinners or load directly.
  navbar_animation: bool, False
      Set navbar is menu transitions should be animated.
  sidebar_state: str, 'auto'
      The starting state of the sidebase, same as variable used in `set_page_config <https://docs.streamlit.io/en/stable/api.html?highlight=set_page_config#streamlit.set_page_config>`.
  hide_streamlit_markers: bool, False
      A flag to hide the default Streamlit menu hamburger button and the footer watermark.
  use_banner_images: str or Array, None
      A path to the image file to use as a banner above the menu or an array of paths to use multiple images spaced using the rations from the banner_spacing parameter.
  banner_spacing: Array, None
      An array to specify the alignment of the banner images, this is the same as the array spec used by Streamlit Columns, if you want centered with 20% padding either side -> banner_spacing =[20,60,20]
  clear_cross_app_sessions: bool, True
      A flag to indicate if the local session store values within individual apps should be cleared when moving to another app, if set to False, when loading sidebar controls, will be a difference between expected and selected.
  session_params: Dict
      A Dict of parameter name and default values that will be added to the global session store, these parameters will be available to all child applications and they can get/set values from the store during execution.

  """

        self._apps = {}
        self._nav_pointers = {}
        self._navbar_pointers = {}
        self._login_app = None
        self._unsecure_app = None
        self._home_app = None
        self._home_label = None
        self._home_id = 'Home'
        self._complex_nav = None
        self._navbar_mode = navbar_mode
        self._navbar_active_index = 0
        self._allow_url_nav = allow_url_nav
        self._navbar_animation = navbar_animation
        self._navbar_sticky = navbar_sticky
        self._nav_item_count = 0
        self._use_navbar = use_navbar
        self._navbar_theme = navbar_theme
        self._banners = use_banner_images
        self._banner_spacing = banner_spacing
        self._hide_streamlit_markers = hide_streamlit_markers
        self._loader_app = LoadingApp()
        self._user_loader = use_loader
        self._use_cookie_cache = use_cookie_cache
        self._cookie_manager = None
        self._logout_label = None
        self._logout_id = 'Logout'
        self._logout_callback = None
        self._login_callback = None
        self._session_attrs = {}
        self._call_queue = []
        self._other_nav = None
        self._guest_name = 'guest'
        self._guest_access = 1
        self._hydralit_url_hash = 'hYDRALIT|-HaShing==seCr8t'
        self._no_access_level = 0

        self._user_session_params = session_params

        try:
            st.set_page_config(page_title=title, page_icon=favicon, layout=layout,
                               initial_sidebar_state=sidebar_state, )
        except:
            pass

        self._nav_horizontal = nav_horizontal

        if self._banners is not None:
            self._banner_container = st.container()

        if nav_container is None:
            self._nav_container = st.container()
        else:
            # hack to stop the beta containers from running set_page_config before HydraApp gets a chance to.
            # if we have a beta_columns container, the instance is delayed until the run() method is called, beta components, who knew!
            if nav_container.__name__ in ['container']:
                self._nav_container = nav_container()
            else:
                self._nav_container = nav_container

        self.cross_session_clear = clear_cross_app_sessions

        if clear_cross_app_sessions:
            preserve_state = 0
        else:
            preserve_state = 1

        if self._user_session_params is None:
            self.session_state = SessionState.get(previous_app=None, selected_app=None, other_nav_app=None,
                                                  preserve_state=preserve_state, allow_access=self._no_access_level,
                                                  logged_in=False, access_hash=None)
            self._session_attrs = {'previous_app': None, 'selected_app': None, 'other_nav_app': None,
                                   'preserve_state': preserve_state, 'allow_access': self._no_access_level,
                                   'logged_in': False, 'access_hash': None}
        else:
            if isinstance(self._user_session_params, Dict):
                self.session_state = SessionState.get(previous_app=None, selected_app=None, other_nav_app=None,
                                                      preserve_state=preserve_state, allow_access=self._no_access_level,
                                                      logged_in=False, current_user=None, access_hash=None,
                                                      **(self._user_session_params))
                self._session_attrs = {'previous_app': None, 'selected_app': None, 'other_nav_app': None,
                                       'preserve_state': preserve_state, 'allow_access': self._no_access_level,
                                       'logged_in': False, 'access_hash': None, **(self._user_session_params)}

    # def _encode_hyauth(self):
    #     user_access_level, username = self.check_access()
    #     payload = {"exp": datetime.now(timezone.utc) + timedelta(days=1), "userid": username,"user_level":user_access_level}
    #     return jwt.encode(payload, self._hydralit_url_hash, algorithm="HS256")

    # def _decode_hyauth(self,token):
    #     return jwt.decode(token, self._hydralit_url_hash, algorithms=["HS256"])

    def add_loader_app(self, loader_app):
        """
  To improve the transition between HydraHeadApps, a loader app is used to quickly clear the window during loading, you can supply a custom loader app, if your include an app that loads a long time to initalise, that is when this app will be seen by the user.
  NOTE: make sure any items displayed are removed once the target app loading is complete, or the items from this app will remain on top of the target app display.
  Parameters
  ------------
  loader_app: HydraHeadApp:`~Hydralit.HydraHeadApp`
      The loader app, this app must implement a modified run method that will receive the target app to be loaded, within the loader run method, the run() method of the target app must be called, or nothing will happen and it will stay in the loader app.
  """

        if loader_app:
            self._loader_app = loader_app
            self._user_loader = True
        else:
            self._loader_app = None
            self._user_loader = False

    def add_app(self, title, app, icon=None, is_login=False, is_home=False, logout_label=None, is_unsecure=False):
        """
  Adds a new application to this HydraApp

  Parameters
  ----------
  title: str
      The title of the app. This is the name that will appear on the menu item for this app.
  app: :HydraHeadApp:`~Hydralit.HydraHeadApp`
      The app class representing the app to include, it must implement inherit from HydraHeadApp classmethod.
  icon: str
      The icon to use on the navigation button, this will be appended to the title to be used on the navigation control.
  is_login: bool, False
      Is this app used to login to the family of apps, this app will provide request response to gateway access to the other apps within the HydraApp.
  is_home: bool, False
      Is this the first 'page' that will be loaded, if a login app is provided, this is the page that will be kicked to upon successful login.
  is_unsecure: bool, False
      An app that can be run other than the login if using security, this is typically a sign-up app that can be run and then kick back to the login.
  """

        # don't add special apps to list
        if self._use_navbar and not is_login and not is_home:
            self._navbar_pointers[title] = [title, icon]

        # if icon is None and not is_login and not is_home:
        #     self._nav_pointers[title] = title
        # else:
        #     self._nav_pointers[title] = '{} {}'.format(icon,title)

        if is_unsecure:
            self._unsecure_app = app

        if is_login:
            self._login_app = app
            self._logout_label = [title, icon]

        elif is_home:
            self._home_app = app
            self._home_label = [title, icon]
        else:
            self._apps[title] = app

        self._nav_item_count = int(self._login_app is not None) + len(self._apps.keys())
        app.assign_session(self.session_state, self)

    def _run_selected(self):
        try:
            if self.session_state.selected_app is None:
                self.session_state.other_nav_app = None
                self.session_state.previous_app = None
                self.session_state.selected_app = self._home_id

                # can disable loader
                if self._user_loader:
                    self._loader_app.run(self._home_app)
                else:
                    self._home_app.run()

                # st.experimental_set_query_params(selected=self._home_app)
            else:

                if self.session_state.other_nav_app is not None:
                    self.session_state.previous_app = self.session_state.selected_app
                    self.session_state.selected_app = self.session_state.other_nav_app
                    self.session_state.other_nav_app = None

                    if self.session_state.selected_app == self._home_id:
                        if self._user_loader:
                            self._loader_app.run(self._home_app)
                        else:
                            self._home_app.run()
                    else:
                        if self._user_loader:
                            self._loader_app.run(self._apps[self.session_state.selected_app])
                        else:
                            self._apps[self.session_state.selected_app].run()
                else:
                    if self.session_state.selected_app == self._home_id:
                        if self._user_loader:
                            self._loader_app.run(self._home_app)
                        else:
                            self._home_app.run()
                    else:
                        if self._user_loader:
                            self._loader_app.run(self._apps[self.session_state.selected_app])
                        else:
                            self._apps[self.session_state.selected_app].run()
                # st.experimental_set_query_params(selected=self.session_state.selected_app)

        except Exception as e:
            st.error('😭 Error triggered from app: **{}**'.format(self.session_state.selected_app))
            st.error('Details: {}'.format(e))

    def _clear_session_values(self):
        for key in st.session_state:
            del st.session_state[key]

    def set_guest(self, guest_name):
        """
  Set the name to be used for guest access.
  Parameters
  -----------
  guest_name: str
      The value to use when allowing guest logins.
  """

        if guest_name is not None:
            self._guest_name = guest_name

    def set_noaccess_level(self, no_access_level: int):
        """
  Set the access level integer value to be used to indicate no access, default is zero.
  Parameters
  -----------
  no_access_level: int
      The value to use to block access, all other values will have some level of access
  """

        if no_access_level is not None:
            self._no_access_level = int(no_access_level)

    def set_access(self, allow_access=0, access_user='', cache_access=False):
        """
  Set the access permission and the assigned username for that access during the current session.
  Parameters
  -----------
  allow_access: int, 0
      Value indicating if access has been granted, can be used to create levels of permission.
  access_user: str, None
      The username the access has been granted to for this session.
  cache_access: bool, False
      Save these access details to a browser cookie so the user will auto login when they visit next time.
  """

        # Set the global access flag
        self.session_state.allow_access = allow_access

        # Also, who are we letting in..
        self.session_state.current_user = access_user

    def check_access(self):
        """
  Check the access permission and the assigned user for the running session.
  Returns
  ---------
  tuple: access_level, username
  """
        username = None

        if hasattr(self.session_state, 'current_user'):
            username = str(self.session_state.current_user)

        return int(self.session_state.allow_access), username

    def get_nav_transition(self):
        """
  Check the previous and current app names the user has navigated between
  Returns
  ---------
  tuple: previous_app, current_app
  """

        return str(self.session_state.previous_app), str(self.session_state.selected_app)

    def get_user_session_params(self):
        """
  Return a dictionary of the keys and current values of the user defined session parameters.
  Returns
  ---------
  dict
  """
        user_session_params = {}

        if self._user_session_params is not None:
            for k in self._user_session_params.keys():
                if hasattr(self.session_state, k):
                    user_session_params[k] = getattr(self.session_state, k)

        return user_session_params

    def _do_logout(self):
        self.session_state.allow_access = self._no_access_level
        self._logged_in = False
        # self._delete_cookie_cache()
        if callable(self._logout_callback):
            self._logout_callback()

        st.experimental_rerun()

    def _run_navbar(self, menu_data):

        if hasattr(hc, '__version__'):

            if hc.__version__ >= 104:
                login_nav = None
                home_nav = None

                if self._login_app:
                    login_nav = {'id': self._logout_id, 'label': self._logout_label[0], 'icon': self._logout_label[1],
                                 'ttip': 'Logout'}

                if self._home_app:
                    home_nav = {'id': self._home_id, 'label': self._home_label[0], 'icon': self._home_label[1],
                                'ttip': 'Home'}

                self.session_state.selected_app = hc.nav_bar(menu_definition=menu_data, key="mainHydralitMenuComplex",
                                                             home_name=home_nav, override_theme=self._navbar_theme,
                                                             login_name=login_nav, use_animation=self._navbar_animation,
                                                             hide_streamlit_markers=self._hide_streamlit_markers)
        else:
            self.session_state.selected_app = hc.nav_bar(menu_definition=menu_data, key="mainHydralitMenuComplex",
                                                         home_name=self._home_app, override_theme=self._navbar_theme,
                                                         login_name=self._logout_label)

        # if nav_selected is not None:
        #     if nav_selected != self.session_state.previous_app and self.session_state.selected_app != nav_selected:
        # self.session_state.selected_app = nav_selected

        if self.cross_session_clear and self.session_state.preserve_state:
            self._clear_session_values()

    def _build_nav_menu(self):

        if self._complex_nav is None:
            number_of_sections = self._nav_item_count
        else:
            number_of_sections = int(self._login_app is not None) + len(self._complex_nav.keys())

        if self._nav_horizontal:
            if hasattr(self._nav_container, 'columns'):
                nav_slots = self._nav_container.columns(number_of_sections)
            elif self._nav_container.__name__ in ['columns']:
                nav_slots = self._nav_container(number_of_sections)
            else:
                nav_slots = self._nav_container
        else:
            if self._nav_container.__name__ in ['columns']:
                # columns within columns currently not supported
                nav_slots = st
            else:
                nav_slots = self._nav_container

        # actually build the menu
        if self._complex_nav is None:
            if self._use_navbar:
                menu_data = [{'label': self._navbar_pointers[app_name][0], 'id': app_name,
                              'icon': self._navbar_pointers[app_name][1]} for app_name in self._apps.keys()]

                # Add logout button and kick to login action
                if self._login_app is not None:
                    # if self.session_state.current_user is not None:
                    #    self._logout_label = '{} : {}'.format(self.session_state.current_user.capitalize(),self._logout_label)

                    with self._nav_container:
                        self._run_navbar(menu_data)

                    # user clicked logout
                    if self.session_state.selected_app == self._logout_label:
                        self._do_logout()
                else:
                    with self._nav_container:
                        self._run_navbar(menu_data)
            else:
                for i, app_name in enumerate(self._apps.keys()):
                    if self._nav_horizontal:
                        nav_section_root = nav_slots[i]
                    else:
                        nav_section_root = nav_slots

                    if nav_section_root.button(label=self._nav_pointers[app_name]):
                        self.session_state.previous_app = self.session_state.selected_app
                        self.session_state.selected_app = app_name

                if self.cross_session_clear and self.session_state.previous_app != self.session_state.selected_app and not self.session_state.preserve_state:
                    self._clear_session_values()

                # Add logout button and kick to login action
                if self._login_app is not None:
                    # if self.session_state.current_user is not None:
                    #    self._logout_label = '{} : {}'.format(self.session_state.current_user.capitalize(),self._logout_label)

                    if self._nav_horizontal:
                        if nav_slots[-1].button(label=self._logout_label):
                            self._do_logout()
                    else:
                        if nav_slots.button(label=self._logout_label):
                            self._do_logout()
        else:
            if self._use_navbar:
                menu_data = []
                for i, nav_section_name in enumerate(self._complex_nav.keys()):
                    menu_item = None
                    if nav_section_name not in [self._home_id, self._logout_id]:
                        if len(self._complex_nav[nav_section_name]) == 1:
                            # if (self._complex_nav[nav_section_name][0] != self._home_app and self._complex_nav[nav_section_name][0] != self._logout_label):
                            menu_item = {'label': self._navbar_pointers[self._complex_nav[nav_section_name][0]][0],
                                         'id': self._complex_nav[nav_section_name][0],
                                         'icon': self._navbar_pointers[self._complex_nav[nav_section_name][0]][1]}
                        else:
                            submenu_items = []
                            for nav_item in self._complex_nav[nav_section_name]:
                                # if (nav_item != self._home_app and nav_item != self._logout_label):
                                menu_item = {'label': self._navbar_pointers[nav_item][0], 'id': nav_item,
                                             'icon': self._navbar_pointers[nav_item][1]}
                                submenu_items.append(menu_item)

                            if len(submenu_items) > 0:
                                menu_item = {'label': nav_section_name, 'id': nav_section_name,
                                             'submenu': submenu_items}

                        if menu_item is not None:
                            menu_data.append(menu_item)

                # Add logout button and kick to login action
                if self._login_app is not None:
                    # if self.session_state.current_user is not None:
                    #    self._logout_label = '{} : {}'.format(self.session_state.current_user.capitalize(),self._logout_label)

                    with self._nav_container:
                        self._run_navbar(menu_data)

                    # user clicked logout
                    if self.session_state.selected_app == self._logout_id:
                        self._do_logout()
                else:
                    # self.session_state.previous_app = self.session_state.selected_app
                    with self._nav_container:
                        self._run_navbar(menu_data)

            else:

                for i, nav_section_name in enumerate(self._complex_nav.keys()):
                    if nav_section_name not in [self._home_id, self._logout_id]:
                        if self._nav_horizontal:
                            nav_section_root = nav_slots[i]
                        else:
                            nav_section_root = nav_slots

                        if len(self._complex_nav[nav_section_name]) == 1:
                            nav_section = nav_section_root.container()
                        else:
                            nav_section = nav_section_root.expander(label=nav_section_name, expanded=False)

                        for nav_item in self._complex_nav[nav_section_name]:
                            if nav_section.button(label=self._nav_pointers[nav_item]):
                                self.session_state.previous_app = self.session_state.selected_app
                                self.session_state.selected_app = nav_item

                if self.cross_session_clear and self.session_state.previous_app != self.session_state.selected_app and not self.session_state.preserve_state:
                    self._clear_session_values()

                    # Add logout button and kick to login action
                if self._login_app is not None:
                    # if self.session_state.current_user is not None:
                    #    self._logout_label = '{} : {}'.format(self.session_state.current_user.capitalize(),self._logout_label)

                    if self._nav_horizontal:
                        if nav_slots[-1].button(label=self._logout_label[0]):
                            self._do_logout()
                    else:
                        if nav_slots.button(label=self._logout_label[0]):
                            self._do_logout()

    def _do_url_params(self):
        if self._allow_url_nav:

            query_params = st.experimental_get_query_params()
            if 'selected' in query_params:
                if (query_params['selected'])[0] != 'None' and (query_params['selected'])[
                    0] != self.session_state.selected_app:  # and (query_params['selected'])[0] != self.session_state.previous_app:
                    self.session_state.other_nav_app = (query_params['selected'])[0]

    def enable_guest_access(self, guest_access_level=1, guest_username='guest'):
        """
  This method will auto login a guest user when the app is secured with a login app, this will allow fora guest user to by-pass the login app and gain access to the other apps that the assigned access level will allow.

  ------------
  guest_access_level: int, 1
      Set the access level to assign to an auto logged in guest user.
  guest_username: str, guest
      Set the username to assign to an auto logged in guest user.
  """

        user_access_level, username = self.check_access()
        if user_access_level == 0 and username is None:
            self.set_access(guest_access_level, guest_username)

    # def get_cookie_manager(self):
    #     if self._use_cookie_cache and self._cookie_manager is not None:
    #         return self._cookie_manager
    #     else:
    #         return None

    # def _delete_cookie_cache(self):
    #     if self._use_cookie_cache and self._cookie_manager is not None:
    #         username_cache = self._cookie_manager.get('hyusername')
    #         accesslevel_cache = self._cookie_manager.get('hyaccesslevel')

    #         if username_cache is not None:
    #             self._cookie_manager.delete('hyusername')

    #         if accesslevel_cache is not None:
    #             self._cookie_manager.delete('hyaccesslevel')

    # def _write_cookie_cache(self,hyaccesslevel,hyusername):
    #     if self._use_cookie_cache and self._cookie_manager is not None:
    #         if hyaccesslevel is not None and hyusername is not None:
    #             self._cookie_manager.set('hyusername',hyusername)
    #             self._cookie_manager.set('hyaccesslevel',hyaccesslevel)

    # def _read_cookie_cache(self):
    #     if self._use_cookie_cache and self._cookie_manager is not None:
    #         username_cache = self._cookie_manager.get('hyusername')
    #         accesslevel_cache = self._cookie_manager.get('hyaccesslevel')

    #         if username_cache is not None and accesslevel_cache is not None:
    #             self.set_access(int(accesslevel_cache), str(username_cache))

    def run(self, complex_nav=None):
        """
  This method is the entry point for the HydraApp, just like a single Streamlit app, you simply setup the additional apps and then call this method to begin.
  Parameters
  ------------
  complex_nav: Dict
      A dictionary that indicates how the nav items should be structured, each key will be a section title and the value will be a list or array of the names of the apps (as registered with the add_app method). The sections with only a single item will be displayed directly, the sections with more than one will be wrapped in an exapnder for cleaner layout.
  """
        # process url navigation parameters
        # self._do_url_params()

        self._complex_nav = complex_nav
        # A hack to hide the hamburger button and Streamlit footer
        # if self._hide_streamlit_markings is not None:
        #    st.markdown(self._hide_streamlit_markings, unsafe_allow_html=True)

        if self._banners is not None:
            if isinstance(self._banners, str):
                self._banners = [self._banners]

            if self._banner_spacing is not None and len(self._banner_spacing) == len(self._banners):
                cols = self._banner_container.columns(self._banner_spacing)
                for idx, im in enumerate(self._banners):
                    if im is not None:
                        if isinstance(im, Dict):
                            cols[idx].markdown(next(iter(im.values())), unsafe_allow_html=True)
                        else:
                            cols[idx].image(im)
            else:
                if self._banner_spacing is not None and len(self._banner_spacing) != len(self._banners):
                    print(
                        'WARNING: Banner spacing spec is a different length to the number of banners supplied, using even spacing for each banner.')

                cols = self._banner_container.columns([1] * len(self._banners))
                for idx, im in enumerate(self._banners):
                    if im is not None:
                        if isinstance(im, Dict):
                            cols[idx].markdown(next(iter(im.values())), unsafe_allow_html=True)
                        else:
                            cols[idx].image(im)

        if self.session_state.allow_access > self._no_access_level or self._login_app is None:
            if callable(self._login_callback):
                if not self.session_state.logged_in:
                    self.session_state.logged_in = True
                    self._login_callback()

            if self._nav_item_count == 0:
                self._default()
            else:
                self._build_nav_menu()
                self._run_selected()
        elif self.session_state.allow_access < self._no_access_level:
            self.session_state.current_user = self._guest_name
            self._unsecure_app.run()
        else:
            self.session_state.logged_in = False
            self.session_state.current_user = None
            self.session_state.access_hash = None
            self._login_app.run()

    def _default(self):
        st.header('Welcome to Hydralit')
        st.write(
            'Thank you for your enthusiasum and looking to run the HydraApp as quickly as possible, for maximum effect, please add a child app by one of the methods below.')

        st.write(
            'For more information, please see the instructions on the home page [Hydralit Home Page](https://github.com/TangleSpace/hydralit)')

        st.write('Method 1 (easiest)')

        st.code("""
#when we import hydralit, we automatically get all of Streamlit
import hydralit as hy

app = hy.HydraApp(title='Simple Multi-Page App')

@app.addapp()
def my_cool_function():
  hy.info('Hello from app 1')
        """
                )

        st.write('Method 2 (more fun)')

        st.code("""
from hydralit import HydraHeadApp
import streamlit as st


#create a child app wrapped in a class with all your code inside the run() method.
class CoolApp(HydraHeadApp):

    def run(self):
        st.info('Hello from cool app 1')



#when we import hydralit, we automatically get all of Streamlit
import hydralit as hy

app = hy.HydraApp(title='Simple Multi-Page App')

app.add_app("My Cool App", icon="📚", app=CoolApp(title="Cool App"))
        """
                )



import hydralit as hy

app = hy.HydraApp(title='Tiko',favicon="🐙",hide_streamlit_markers=True,use_navbar=True, navbar_sticky=True, use_loader=True)

#we have added a sign-up app to demonstrate the ability to run an unsecure app
    #only 1 unsecure app is allowed

# we have added a sign-up app to demonstrate the ability to run an unsecure app
# only 1 unsecure app is allowed
app.add_app("Signup", icon="🛰️", app=apps.SignUpApp(title='Signup'), is_unsecure=True)
app.add_app("WalshApp", icon="🛰️", app=apps.WalshApp(title='WalshApp'), is_unsecure=True)
app.add_app("WalshAppSecure", icon="🛰️", app=apps.WalshAppSecure(title='WalshApp Secure'), is_unsecure=True)


# we want to have secure access for this HydraApp, so we provide a login application
# optional logout label, can be blank for something nicer!
app.add_app("Login", apps.LoginApp(title='Login'), is_login=True)

# specify a custom loading app for a custom transition between apps, this includes a nice custom spinner
app.add_loader_app(apps.MyLoadingApp(delay=3))




# app.add_loader_app(apps.QuickLoaderApp())

# we can inject a method to be called everytime a user logs out
@app.logout_callback
def mylogout_cb():
    st.write('I was called from Hydralit at logout!')


# we can inject a method to be called everytime a user logs in
@app.login_callback
def mylogin_cb():
    st.write('I was called from Hydralit at login!')


# if we want to auto login a guest but still have a secure app, we can assign a guest account and go straight in
app.enable_guest_access()
# check user access level to determine what should be shown on the menu
user_access_level, username = app.check_access()

@app.addapp(is_home=True)
def my_home():
   hy.info('Hello from Home!')

@app.addapp(title='Forex')
def forex():
    from tradingview_ta import TA_Handler, Interval
    import streamlit as st
    from datetime import date
    today = date.today()
    import tradingview_ta, requests, os
    from datetime import timezone
    from st_aggrid import AgGrid
    import pandas as pd
    import yfinance as yf
    from streamlit_echarts import st_echarts
    import matplotlib.pyplot as plt

    def color_negative_red(val):
        color = 'red' if val == 'SELL' else 'green' if val == 'BUY' else 'green' if val == 'STRONG_BUY' else 'red' if val == 'STRONG_SELL' else 'white'
        return 'color: %s' % color

    def get_analysis(symbol: str, exchange: str, screener: str, interval: str):
        # TradingView Technical Analysis
        handler = tradingview_ta.TA_Handler()
        handler.set_symbol_as(symbol)
        handler.set_exchange_as_crypto_or_stock(exchange)
        handler.set_screener_as_stock(screener)
        handler.set_interval_as(interval)
        analysis = handler.get_analysis()

        return analysis

    st.title("Forex Trading Signals")

    col25, col26 = st.columns(2)
    with col25:
        currency_pair = pd.read_csv(
            '/Users/administrator/PycharmProjects/MultipleApp/financialdashboard/pages/currency_pair.csv')
        symbol = st.selectbox('Select the Currency Pairs', currency_pair).upper()

    exchange = "FX_IDC"
    screener = "forex"
    with col26:
        interval = st.selectbox("Interval", ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1W", "1M"))

    tesla = TA_Handler()
    tesla.set_symbol_as(symbol)
    tesla.set_exchange_as_crypto_or_stock(exchange)
    tesla.set_screener_as_stock(screener)
    tesla.set_interval_as(interval)

    analysis = get_analysis(symbol, exchange, screener, interval)
    st.markdown("Success!")


    # st.title("Symbol: `" + analysis.symbol + "`")

    # st.markdown("Exchange: `" + analysis.exchange + "`")
    # st.markdown("Screener: `" + analysis.screener + "`")

    # st.header("Interval: `" + analysis.interval + "`")

    # if analysis.time and analysis.time.astimezone():
    # st.markdown("Time (UTC): `" + str(analysis.time.astimezone(timezone.utc)) + "`")


    col14, col15 = st.columns(2)
    with col14:
        st.header("Symbol: `" + analysis.symbol + "`")
    with col15:
        st.header("Interval: `" + analysis.interval + "`")

    st.header("Summary Of Indicators")
    if analysis.time and analysis.time.astimezone():
        st.markdown("Time (UTC): `" + str(analysis.time.astimezone(timezone.utc)) + "`")

    a = tesla.get_analysis().summary

    col10, col12, col13 = st.columns(3)
    col10.metric('RECOMMENDATION', a['RECOMMENDATION'])
    col12.metric("BUY", a['BUY'])
    col13.metric("SELL", a['SELL'])

    tesla1 = TA_Handler()
    tesla1.set_symbol_as(symbol)
    tesla1.set_exchange_as_crypto_or_stock(exchange)
    tesla1.set_screener_as_stock(screener)
    tesla1.set_interval_as("1m")

    tesla2 = TA_Handler()
    tesla2.set_symbol_as(symbol)
    tesla2.set_exchange_as_crypto_or_stock(exchange)
    tesla2.set_screener_as_stock(screener)
    tesla2.set_interval_as("5m")

    tesla3 = TA_Handler()
    tesla3.set_symbol_as(symbol)
    tesla3.set_exchange_as_crypto_or_stock(exchange)
    tesla3.set_screener_as_stock(screener)
    tesla3.set_interval_as("15m")

    tesla4 = TA_Handler()
    tesla4.set_symbol_as(symbol)
    tesla4.set_exchange_as_crypto_or_stock(exchange)
    tesla4.set_screener_as_stock(screener)
    tesla4.set_interval_as("30m")

    tesla5 = TA_Handler()
    tesla5.set_symbol_as(symbol)
    tesla5.set_exchange_as_crypto_or_stock(exchange)
    tesla5.set_screener_as_stock(screener)
    tesla5.set_interval_as("1h")

    tesla6 = TA_Handler()
    tesla6.set_symbol_as(symbol)
    tesla6.set_exchange_as_crypto_or_stock(exchange)
    tesla6.set_screener_as_stock(screener)
    tesla6.set_interval_as("4h")

    tesla7 = TA_Handler()
    tesla7.set_symbol_as(symbol)
    tesla7.set_exchange_as_crypto_or_stock(exchange)
    tesla7.set_screener_as_stock(screener)
    tesla7.set_interval_as("1d")

    tesla8 = TA_Handler()
    tesla8.set_symbol_as(symbol)
    tesla8.set_exchange_as_crypto_or_stock(exchange)
    tesla8.set_screener_as_stock(screener)
    tesla8.set_interval_as("1W")

    tesla9 = TA_Handler()
    tesla9.set_symbol_as(symbol)
    tesla9.set_exchange_as_crypto_or_stock(exchange)
    tesla9.set_screener_as_stock(screener)
    tesla9.set_interval_as("1M")

    my_expander1 = st.expander(label='Real Time Recommendation - Expand me')
    with my_expander1:
        cola, col0, col1, col2 = st.columns(4)
        cola.header("Date")
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        col0.header("Currency Pair")
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col1.header("TimeFrame")
        col1.write("1 Minute")
        col1.write("5 Minutes")
        col1.write("15 Minutes")
        col1.write("30 Minute")
        col1.write("1 Hour")
        col1.write("4 Hours")
        col1.write("1 Day")
        col1.write("1 Week")
        col1.write("1 Month")
        col2.header("Signals")
        col2.write(tesla1.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla2.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla3.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla4.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla5.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla6.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla7.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla8.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla9.get_analysis().summary["RECOMMENDATION"])

    b = tesla.get_analysis().oscillators
    df1 = pd.DataFrame(b)
    df1 = df1.style.applymap(color_negative_red)

    d = tesla.get_analysis().moving_averages
    df4 = pd.DataFrame(d)
    df4 = df4.style.applymap(color_negative_red)

    my_expander = st.expander(label='Indicators - Expand me')
    with my_expander:
        col1, col2 = st.columns(2)
        col1.header("Moving Averages")
        col1.table(df4)
        col2.header("Oscillators")
        col2.table(df1)

    my_expander2 = st.expander(label='Live Historical Data - Expand me')
    tick = (symbol + "=X")

    data = yf.download(tickers=tick, period="1d", interval=interval)
    df = pd.DataFrame(data)
    df = df.reset_index()
    df = df.set_index('Datetime')

    with my_expander2:
        st.dataframe(df)

    my_expander3 = st.expander(label='Chart - Expand me')
    with my_expander3:
        st.line_chart(df["Close"])

        st.write("")

        st.bar_chart(df["Close"])

@app.addapp(title='Crypto Prediction')
def Cryptoprediction():
    # Prophet is basically a forecasting model that can be used to predict future values of time series data.
    # yahoo finance to get the crypto data
    import streamlit as st
    from datetime import date

    import yfinance as yf
    from fbprophet import Prophet

    from fbprophet.plot import plot_plotly
    from plotly import graph_objs as go

    START = "2015-01-01"  # start date
    TODAY = date.today().strftime("%Y-%m-%d")  # today's date and convert it into string format

    st.title("Crypto Prediction App")  # title of the app

    cryptos = ("BTC-USD", "ETH-USD", "BNB-USD", "USDT-USD")  # list of cryptos i.e. apple, google, microsoft, GameStop

    selected_cryptos = st.selectbox("Select dataset for prediction",
                                    cryptos)  # select the crypto from the select box and return it into selected_cryptos variable

    n_years = st.slider("Years Of prediction:", 1, 4)  # slider to select the number of years of prediction from 1 to 4

    period = n_years * 365  # number of days in a year

    @st.cache  # store the donwloaded data in the cache so we don't have to download it again
    def load_data(ticker):  # ticker is basically the selected_cryptos variable
        data = yf.download(ticker, START,
                           TODAY)  # to download the data from yahoo finance for a specified start and end date
        data.reset_index(inplace=True)  # This puts date in the very first column
        return data

    data_load_state = st.text("Loading data...")  # text to show the loading of data

    data = load_data(selected_cryptos)  # synchronous call to function
    data_load_state.text("Loading data...done!")  # after the data is loaded, we can plot the raw data

    st.subheader('Raw Data')  # subheader for raw data
    st.write(data.tail())  # display the last 5 rows of the data

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='crypto_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='crypto_close'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Forecasting()
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds",
                                        "Close": "y"})  # we have to rename because prophet expects the column names to be ds and y

    m = Prophet()  # Creating a model
    m.fit(df_train)  # fitting the training data

    future = m.make_future_dataframe(
        periods=period)  # dataframe for the future data ; period is the number of days we want to predict
    forecast = m.predict(future)  # predicting the future data

    st.subheader('Forecast data')  # subheader for forecast data
    st.write(forecast.tail())  # display the last 5 rows of the forecast data

    st.write('forecast data')  # display the forecast data
    fig1 = plot_plotly(m, forecast)  # plot the forecast data
    st.plotly_chart(fig1)  # plot the forecast data

    st.write('forecast components')  # display the forecast components
    fig2 = m.plot_components(forecast)  # plot the forecast components
    st.write(fig2)  # plot the forecast components


@app.addapp(title='Crypto Signal')
def Crypto():
    from tradingview_ta import TA_Handler, Interval
    import streamlit as st
    import tradingview_ta, requests, os
    from datetime import timezone
    from datetime import date
    today = date.today()
    from st_aggrid import AgGrid
    import pandas as pd
    import yfinance as yf
    from streamlit_echarts import st_echarts
    import matplotlib.pyplot as plt

    def color_negative_red(val):
        color = 'red' if val == 'SELL' else 'green' if val == 'BUY' else 'green' if val == 'STRONG_BUY' else 'red' if val == 'STRONG_SELL' else 'white'
        return 'color: %s' % color

    def get_analysis(symbol: str, exchange: str, screener: str, interval: str):
        # TradingView Technical Analysis
        handler = tradingview_ta.TA_Handler()
        handler.set_symbol_as(symbol)
        handler.set_exchange_as_crypto_or_stock(exchange)
        handler.set_screener_as_stock(screener)
        handler.set_interval_as(interval)
        analysis = handler.get_analysis()

        return analysis

    st.title("CryptoCurrency Trading Signals")

    currency_pair = pd.read_csv(
        '/Users/administrator/PycharmProjects/MultipleApp/financialdashboard/hydralit-example-main/test-hydralit-working/crypto2.csv')

    symbol = st.selectbox('Select the Currency Pairs', currency_pair).upper()

    exchange = st.selectbox("Exchange", (
    "BINANCE", "COINBASE", "COINEX", "BINGX", "KUCOIN", "UNISWAP", "POLONIEX", "BITTREX", "HUOBI"))
    screener = "crypto"

    interval = st.selectbox("Interval", ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1W", "1M"))

    tesla = TA_Handler()
    tesla.set_symbol_as(symbol)
    tesla.set_exchange_as_crypto_or_stock(exchange)
    tesla.set_screener_as_stock(screener)
    tesla.set_interval_as(interval)

    analysis = get_analysis(symbol, exchange, screener, interval)
    st.markdown("Success!")


    col14, col15 = st.columns(2)
    with col14:
        st.title("Symbol: `" + analysis.symbol + "`")
    with col15:
        st.title("Interval: `" + analysis.interval + "`")

    a = tesla.get_analysis().summary

    st.header("Summary Of Indicators")
    if analysis.time and analysis.time.astimezone():
        st.markdown("Time (UTC): `" + str(analysis.time.astimezone(timezone.utc)) + "`")

    col10, col12, col13 = st.columns(3)
    col10.metric('RECOMMENDATION', a['RECOMMENDATION'])
    col12.metric("BUY", a['BUY'])
    col13.metric("SELL", a['SELL'])

    tesla1 = TA_Handler()
    tesla1.set_symbol_as(symbol)
    tesla1.set_exchange_as_crypto_or_stock(exchange)
    tesla1.set_screener_as_stock(screener)
    tesla1.set_interval_as("1m")

    tesla2 = TA_Handler()
    tesla2.set_symbol_as(symbol)
    tesla2.set_exchange_as_crypto_or_stock(exchange)
    tesla2.set_screener_as_stock(screener)
    tesla2.set_interval_as("5m")

    tesla3 = TA_Handler()
    tesla3.set_symbol_as(symbol)
    tesla3.set_exchange_as_crypto_or_stock(exchange)
    tesla3.set_screener_as_stock(screener)
    tesla3.set_interval_as("15m")

    tesla4 = TA_Handler()
    tesla4.set_symbol_as(symbol)
    tesla4.set_exchange_as_crypto_or_stock(exchange)
    tesla4.set_screener_as_stock(screener)
    tesla4.set_interval_as("30m")

    tesla5 = TA_Handler()
    tesla5.set_symbol_as(symbol)
    tesla5.set_exchange_as_crypto_or_stock(exchange)
    tesla5.set_screener_as_stock(screener)
    tesla5.set_interval_as("1h")

    tesla6 = TA_Handler()
    tesla6.set_symbol_as(symbol)
    tesla6.set_exchange_as_crypto_or_stock(exchange)
    tesla6.set_screener_as_stock(screener)
    tesla6.set_interval_as("4h")

    tesla7 = TA_Handler()
    tesla7.set_symbol_as(symbol)
    tesla7.set_exchange_as_crypto_or_stock(exchange)
    tesla7.set_screener_as_stock(screener)
    tesla7.set_interval_as("1d")

    tesla8 = TA_Handler()
    tesla8.set_symbol_as(symbol)
    tesla8.set_exchange_as_crypto_or_stock(exchange)
    tesla8.set_screener_as_stock(screener)
    tesla8.set_interval_as("1W")

    tesla9 = TA_Handler()
    tesla9.set_symbol_as(symbol)
    tesla9.set_exchange_as_crypto_or_stock(exchange)
    tesla9.set_screener_as_stock(screener)
    tesla9.set_interval_as("1M")

    my_expander1 = st.expander(label='Real Time Recommendation - Expand me')
    with my_expander1:
        cola, col0, col1, col2 = st.columns(4)
        cola.header("Date")
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        cola.write(today)
        col0.header("Currency Pair")
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col0.write(symbol)
        col1.header("TimeFrame")
        col1.write("1 Minute")
        col1.write("5 Minutes")
        col1.write("15 Minutes")
        col1.write("30 Minute")
        col1.write("1 Hour")
        col1.write("4 Hours")
        col1.write("1 Day")
        col1.write("1 Week")
        col1.write("1 Month")
        col2.header("Signals")
        col2.write(tesla1.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla2.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla3.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla4.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla5.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla6.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla7.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla8.get_analysis().summary["RECOMMENDATION"])
        col2.write(tesla9.get_analysis().summary["RECOMMENDATION"])

    b = tesla.get_analysis().oscillators
    df1 = pd.DataFrame(b)
    df1 = df1.style.applymap(color_negative_red)

    d = tesla.get_analysis().moving_averages
    df4 = pd.DataFrame(d)
    df4 = df4.style.applymap(color_negative_red)

    my_expander = st.expander(label='Indicators - Expand me')
    with my_expander:
        col1, col2 = st.columns(2)
        col1.header("Moving Averages")
        col1.table(df4)
        col2.header("Oscillators")
        col2.table(df1)

    my_expander2 = st.expander(label='Live Historical Data - Expand me')
    tick = (symbol + "=X")

    data = yf.download(tickers='BTC-USD', period='1d', interval=interval)
    df = pd.DataFrame(data)
    df = df.reset_index()
    df = df.set_index('Datetime')

    with my_expander2:
        st.write(df)

    my_expander3 = st.expander(label='Chart - Expand me')
    with my_expander3:

        st.line_chart(df["Close"])

        st.write("")

        st.bar_chart(df["Close"])


@app.addapp(title='Stock Signals')
def stocksignals():
           if user_access_level > 0:
               st.title('Stocks Signals - Over 8000 Stocks')



               df = pd.read_csv(
                   "/Users/administrator/PycharmProjects/MultipleApp/financialdashboard/hydralit-example-main/test-hydralit-working/hydralit-update/hydralit/StockSymbolLists-main/AMEX.csv")

               AMEX = df['Symbol'].tolist()


               df1 = pd.read_csv(
                   "/Users/administrator/PycharmProjects/MultipleApp/financialdashboard/hydralit-example-main/test-hydralit-working/hydralit-update/hydralit/StockSymbolLists-main/NYSE.csv")

               NYSE = df1['Symbol'].tolist()

               df2 = pd.read_csv("/Users/administrator/PycharmProjects/MultipleApp/financialdashboard/hydralit-example-main/test-hydralit-working/hydralit-update/hydralit/StockSymbolLists-main/NASDAQ.csv")
               NASDAQ = df2['Symbol'].tolist()

               col31, col32, col33 = st.columns(3)
               with col31:

                  option = st.selectbox('Select Exchange', ['NYSE', 'NASDAQ', 'AMEX'])

               if option == 'NASDAQ':
                   symbol = df2['Symbol'].tolist()
                   exchange = 'NASDAQ'
               elif option == 'NYSE':
                   symbol = df1['Symbol'].tolist()
                   exchange = 'NYSE'
               elif option == 'AMEX':
                    symbol = df['Symbol'].tolist()
                    exchange = 'AMEX'
               with col32:

                  symbol = st.selectbox("Select Symbol", symbol)
                  ticker = symbol





               def color_negative_red(val):
                   color = 'red' if val == 'SELL' else 'green' if val == 'BUY' else 'green' if val == 'STRONG_BUY' else 'red' if val == 'STRONG_SELL' else 'white'
                   return 'color: %s' % color

               def get_analysis(symbol: str, exchange: str, screener: str, interval: str):
                   # TradingView Technical Analysis
                   handler = tradingview_ta.TA_Handler()
                   handler.set_symbol_as(symbol)
                   handler.set_exchange_as_crypto_or_stock(exchange)
                   handler.set_screener_as_stock(screener)
                   handler.set_interval_as(interval)
                   analysis = handler.get_analysis()

                   return analysis


               screener = "america"
               with col33:

                   interval = st.selectbox("Interval", ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1W", "1M"))

               tesla = TA_Handler()
               tesla.set_symbol_as(symbol)
               tesla.set_exchange_as_crypto_or_stock(exchange)
               tesla.set_screener_as_stock(screener)
               tesla.set_interval_as(interval)

               analysis = get_analysis(symbol, exchange, screener, interval)
               st.markdown("Success!")
               name = yahooFinance.Ticker(symbol).info['shortName']
               col13, col15, col16 = st.columns(3)
               with col13:
                   st.title(name)


               with col15:
                   st.title("Symbol: `" + analysis.symbol + "`")

               with col16:
                   st.title("Interval: `" + analysis.interval + "`")


               a = tesla.get_analysis().summary

               my_expander = st.expander(label= name + 'Company Keys Values - Expand me')
               with my_expander:


                   GetFacebookInformation = yahooFinance.Ticker(symbol)
                   # get all key value pairs that are available
                   for key, value in GetFacebookInformation.info.items():
                       st.write(key, ":", value)

                   def get_yahoo_shortname(symbol):
                       response = urllib.request.urlopen(
                           f'https://query2.finance.yahoo.com/v1/finance/search?q={symbol}')
                       content = response.read()
                       data = json.loads(content.decode('utf8'))['quotes'][0]['shortname']
                       return data



               st.header("Summary Of Indicators")
               if analysis.time and analysis.time.astimezone():
                   st.markdown("Time (UTC): `" + str(analysis.time.astimezone(timezone.utc)) + "`")

               col10, col12, col13 = st.columns(3)
               col10.metric('RECOMMENDATION', a['RECOMMENDATION'])
               col12.metric("BUY", a['BUY'])
               col13.metric("SELL", a['SELL'])

               tesla1 = TA_Handler()
               tesla1.set_symbol_as(symbol)
               tesla1.set_exchange_as_crypto_or_stock(exchange)
               tesla1.set_screener_as_stock(screener)
               tesla1.set_interval_as("1m")

               tesla2 = TA_Handler()
               tesla2.set_symbol_as(symbol)
               tesla2.set_exchange_as_crypto_or_stock(exchange)
               tesla2.set_screener_as_stock(screener)
               tesla2.set_interval_as("5m")

               tesla3 = TA_Handler()
               tesla3.set_symbol_as(symbol)
               tesla3.set_exchange_as_crypto_or_stock(exchange)
               tesla3.set_screener_as_stock(screener)
               tesla3.set_interval_as("15m")

               tesla4 = TA_Handler()
               tesla4.set_symbol_as(symbol)
               tesla4.set_exchange_as_crypto_or_stock(exchange)
               tesla4.set_screener_as_stock(screener)
               tesla4.set_interval_as("30m")

               tesla5 = TA_Handler()
               tesla5.set_symbol_as(symbol)
               tesla5.set_exchange_as_crypto_or_stock(exchange)
               tesla5.set_screener_as_stock(screener)
               tesla5.set_interval_as("1h")

               tesla6 = TA_Handler()
               tesla6.set_symbol_as(symbol)
               tesla6.set_exchange_as_crypto_or_stock(exchange)
               tesla6.set_screener_as_stock(screener)
               tesla6.set_interval_as("4h")

               tesla7 = TA_Handler()
               tesla7.set_symbol_as(symbol)
               tesla7.set_exchange_as_crypto_or_stock(exchange)
               tesla7.set_screener_as_stock(screener)
               tesla7.set_interval_as("1d")

               tesla8 = TA_Handler()
               tesla8.set_symbol_as(symbol)
               tesla8.set_exchange_as_crypto_or_stock(exchange)
               tesla8.set_screener_as_stock(screener)
               tesla8.set_interval_as("1W")

               tesla9 = TA_Handler()
               tesla9.set_symbol_as(symbol)
               tesla9.set_exchange_as_crypto_or_stock(exchange)
               tesla9.set_screener_as_stock(screener)
               tesla9.set_interval_as("1M")

               my_expander1 = st.expander(label='Real Time Recommendation - Expand me')
               with my_expander1:
                   cola, col0, col1, col2 = st.columns(4)
                   cola.header("Date")
                   cola.write(today)
                   cola.write(today)
                   cola.write(today)
                   cola.write(today)
                   cola.write(today)
                   cola.write(today)
                   cola.write(today)
                   cola.write(today)
                   cola.write(today)
                   col0.header("Currency Pair")
                   col0.write(symbol)
                   col0.write(symbol)
                   col0.write(symbol)
                   col0.write(symbol)
                   col0.write(symbol)
                   col0.write(symbol)
                   col0.write(symbol)
                   col0.write(symbol)
                   col0.write(symbol)
                   col1.header("TimeFrame")
                   col1.write("1 Minute")
                   col1.write("5 Minutes")
                   col1.write("15 Minutes")
                   col1.write("30 Minute")
                   col1.write("1 Hour")
                   col1.write("4 Hours")
                   col1.write("1 Day")
                   col1.write("1 Week")
                   col1.write("1 Month")
                   col2.header("Signals")
                   col2.write(tesla1.get_analysis().summary["RECOMMENDATION"])
                   col2.write(tesla2.get_analysis().summary["RECOMMENDATION"])
                   col2.write(tesla3.get_analysis().summary["RECOMMENDATION"])
                   col2.write(tesla4.get_analysis().summary["RECOMMENDATION"])
                   col2.write(tesla5.get_analysis().summary["RECOMMENDATION"])
                   col2.write(tesla6.get_analysis().summary["RECOMMENDATION"])
                   col2.write(tesla7.get_analysis().summary["RECOMMENDATION"])
                   col2.write(tesla8.get_analysis().summary["RECOMMENDATION"])
                   col2.write(tesla9.get_analysis().summary["RECOMMENDATION"])

               b = tesla.get_analysis().oscillators
               df1 = pd.DataFrame(b)
               df1 = df1.style.applymap(color_negative_red)

               d = tesla.get_analysis().moving_averages
               df4 = pd.DataFrame(d)
               df4 = df4.style.applymap(color_negative_red)

               my_expander = st.expander(label='Indicators - Expand me')
               with my_expander:
                   col1, col2 = st.columns(2)
                   col1.header("Moving Averages")
                   col1.table(df4)
                   col2.header("Oscillators")
                   col2.table(df1)

               my_expander2 = st.expander(label='Live Historical Data - Expand me')

               data = yf.download(tickers=symbol, period='1d', interval='1m')
               df = pd.DataFrame(data)
               df = df.reset_index()
               df = df.set_index('Datetime')

               with my_expander2:
                   st.write(df)

               my_expander3 = st.expander(label='Chart - Expand me')
               with my_expander3:

                   st.line_chart(df["Close"])

                   st.write("")

                   st.bar_chart(df["Close"])


           else:
               st.write("consider upgrading your account")

@app.addapp(title='Backtesting')
def backtesting():
    st.title('Backtesting Dashboard.')
    st.markdown('Backtest different trading strategies on 4000+ US Stocks')

    st.sidebar.title('Stock Backtesting')

    st.sidebar.markdown('')

    scripts = import_scripts()
    indicators = import_indicators()

    backtest_timeframe = st.sidebar.expander('BACKTEST TIMEFRAME')

    start_date = backtest_timeframe.date_input('Starting Date', value=dt(2017, 1, 1), min_value=dt(2015, 1, 1),
                                               max_value=dt(2019, 1, 1))
    str_start_date = str(start_date)
    start_date = str_start_date[-2:] + '/' + str_start_date[5:7] + '/' + str_start_date[:4]

    end_date = backtest_timeframe.date_input('Ending Date', min_value=dt(2021, 1, 10))
    str_end_date = str(end_date)
    end_date = str_end_date[-2:] + '/' + str_end_date[5:7] + '/' + str_end_date[:4]

    symbol = st.selectbox('Stock Name', scripts)
    ticker = str(symbol).split('(')[1][:-1]

    indicator = st.selectbox('INDICATOR', indicators)

    df_start = dt(2013, 1, 1)
    str_dfstart_date = str(df_start)[:-9]
    df_start = str_dfstart_date[-2:] + '/' + str_dfstart_date[5:7] + '/' + str_dfstart_date[:4]

    data = inv.get_stock_historical_data(stock=ticker, country='United States', from_date=df_start,
                                         to_date=end_date)

    # 1. SUPERTREND
    if indicator == 'SuperTrend':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_supertrend(
            st, data, start_date, end_date)

    # 2. -DI, NEGATIVE DIRECTIONAL INDEX
    if indicator == '-DI, Negative Directional Index':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_negative_directional_index(
            st, data, start_date, end_date)

    # 3. NORMALIZED AVERAGE TRUE RANGE
    if indicator == 'Normalized Average True Range (NATR)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_normalized_average_true_range(
            st, data, start_date, end_date)

    # 4. AVERAGE DIRECTIONAL INDEX
    if indicator == 'Average Directional Index (ADX)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_average_directional_index(
            st, data, start_date, end_date)

    # 5. STOCHASTIC OSCILLATOR FAST
    if indicator == 'Stochastic Oscillator Fast (SOF)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_stochastic_oscillator_fast(
            st, data, start_date, end_date)

    # 6. STOCHASTIC OSCILLATOR SLOW
    if indicator == 'Stochastic Oscillator Slow (SOS)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_stochastic_oscillator_slow(
            st, data, start_date, end_date)

    # 7. WEIGHTED MOVING AVERAGE
    if indicator == 'Weighted Moving Average (WMA)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_weighted_moving_average(
            st, data, start_date, end_date)

    # 8. MOMENTUM INDICATOR
    if indicator == 'Momentum Indicator (MOM)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_momentum_indicator(
            st, data, start_date, end_date)

    # 7. VORTEX INDICATOR
    if indicator == 'Vortex Indicator (VI)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_vortex_indicator(
            st, data, start_date, end_date)

    # 8. CHANDE MOMENTUM OSCILLATOR
    if indicator == 'Chande Momentum Oscillator (CMO)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_chande_momentum_oscillator(
            st, data, start_date, end_date)

    # 9. EXPONENTIAL MOVING AVERAGE
    if indicator == 'Exponential Moving Average (EMA)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_exponential_moving_average(
            st, data, start_date, end_date)

    # 10. TRIPLE EXPONENTIAL MOVING AVERAGE
    if indicator == 'Triple Exponential Moving Average (TEMA)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_triple_exponential_moving_average(
            st, data, start_date, end_date)

    # 11. DOUBLE EXPONENTIAL MOVING AVERAGE
    if indicator == 'Double Exponential Moving Average (DEMA)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_double_exponential_moving_average(
            st, data, start_date, end_date)

    # 12. SIMPLE MOVING AVERAGE
    if indicator == 'Simple Moving Average (SMA)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_simple_moving_average(
            st, data, start_date, end_date)

    # 13. TRIANGULAR MOVING AVERAGE
    if indicator == 'Triangular Moving Average (TRIMA)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_triangular_moving_average(
            st, data, start_date, end_date)

    # 14. CHANDE FORECAST OSCILLATOR
    if indicator == 'Chande Forecast Oscillator (CFO)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_chande_forecast_oscillator(
            st, data, start_date, end_date)

    # 15. CHOPPINESS INDEX
    if indicator == 'Choppiness Index':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_choppiness_index(
            st, data, start_date, end_date)

    # 16. AROON DOWN
    if indicator == 'Aroon Down':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_aroon_down(
            st, data, start_date, end_date)

    # 16. AVERAGE TRUE RANGE
    if indicator == 'Average True Range (ATR)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_average_true_range(
            st, data, start_date, end_date)

    # 17. WILLIAMS %R
    if indicator == 'Williams %R':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_williamsr(
            st, data, start_date, end_date)

    # 18. PARABOLIC SAR
    if indicator == 'Parabolic SAR':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_parabolic_sar(
            st, data, start_date, end_date)

    # 19. COPPOCK CURVE
    if indicator == 'Coppock Curve':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_coppock_curve(
            st, data, start_date, end_date)

    # 20. +DI, POSITIVE DIRECTIONAL INDEX
    if indicator == '+DI, Positive Directional Index':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_positive_directional_index(
            st, data, start_date, end_date)

    # 21. RELATIVE STRENGTH INDEX
    if indicator == 'Relative Strength Index (RSI)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_rsi(st,
                                                                                                            data,
                                                                                                            start_date,
                                                                                                            end_date)

    # 22. MACD Signal
    if indicator == 'MACD Signal':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_macd_signal(
            st, data, start_date, end_date)

    # 23. AROON OSCILLATOR
    if indicator == 'Aroon Oscillator':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_aroon_oscillator(
            st, data, start_date, end_date)

    # 24. STOCHASTIC RSI FASTK
    if indicator == 'Stochastic RSI FastK':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_stochrsi_fastk(
            st, data, start_date, end_date)

    # 25. STOCHASTIC RSI FASTD
    if indicator == 'Stochastic RSI FastD':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_stochrsi_fastd(
            st, data, start_date, end_date)

    # 26. ULTIMATE OSCILLATOR
    if indicator == 'Ultimate Oscillator':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_ultimate_oscillator(
            st, data, start_date, end_date)

    # 27. AROON UP
    if indicator == 'Aroon Up':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_aroon_up(st,
                                                                                                                 data,
                                                                                                                 start_date,
                                                                                                                 end_date)

    # 28. BOLLINGER BANDS
    if indicator == 'Bollinger Bands':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_bollinger_bands(
            st, data, start_date, end_date)

    # 29. TRIX
    if indicator == 'TRIX':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_trix(st,
                                                                                                             data,
                                                                                                             start_date,
                                                                                                             end_date)

    # 30. COMMODITY CHANNEL INDEX
    if indicator == 'Commodity Channel Index (CCI)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_cci(st,
                                                                                                            data,
                                                                                                            start_date,
                                                                                                            end_date)

    # 31. MACD
    if indicator == 'MACD':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_macd(st,
                                                                                                             data,
                                                                                                             start_date,
                                                                                                             end_date)

    # 31. MACD HISTOGRAM
    if indicator == 'MACD Histogram':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_macd_histogram(
            st, data, start_date, end_date)

    # 32. MONEY FLOW INDEX
    if indicator == 'Money Flow Index (MFI)':
        entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_mfi(st,
                                                                                                            data,
                                                                                                            start_date,
                                                                                                            end_date)

    st.sidebar.markdown('')
    cf_bt = st.sidebar.button('Save & Backtest')
    if cf_bt == False:
        st.info('Hit the "Save & Backtest" button at the bottom left corner to view the results')
    elif cf_bt == True:
        backtestdata = inv.get_stock_historical_data(stock=ticker, country='United States', from_date=start_date,
                                                     to_date=end_date)
        if entry_comparator == '<, Crossing Down' and exit_comparator == '<, Crossing Down':
            buy_price, sell_price, strategy_signals = crossingdown_crossingdown(backtestdata, entry_data1,
                                                                                entry_data2, exit_data1, exit_data2)
        elif entry_comparator == '<, Crossing Down' and exit_comparator == '>, Crossing Up':
            buy_price, sell_price, strategy_signals = crossingdown_crossingup(backtestdata, entry_data1,
                                                                              entry_data2, exit_data1, exit_data2)
        elif entry_comparator == '<, Crossing Down' and exit_comparator == '==, Equal To':
            buy_price, sell_price, strategy_signals = crossingdown_equalto(backtestdata, entry_data1, entry_data2,
                                                                           exit_data1, exit_data2)
        elif entry_comparator == '>, Crossing Up' and exit_comparator == '<, Crossing Down':
            buy_price, sell_price, strategy_signals = crossingup_crossingdown(backtestdata, entry_data1,
                                                                              entry_data2, exit_data1, exit_data2)
        elif entry_comparator == '>, Crossing Up' and exit_comparator == '>, Crossing Up':
            buy_price, sell_price, strategy_signals = crossingup_crossingup(backtestdata, entry_data1, entry_data2,
                                                                            exit_data1, exit_data2)
        elif entry_comparator == '>, Crossing Up' and exit_comparator == '==, Equal To':
            buy_price, sell_price, strategy_signals = crossingup_equalto(backtestdata, entry_data1, entry_data2,
                                                                         exit_data1, exit_data2)
        elif entry_comparator == '==, Equal To' and exit_comparator == '>, Crossing Up':
            buy_price, sell_price, strategy_signals = equalto_crossingup(backtestdata, entry_data1, entry_data2,
                                                                         exit_data1, exit_data2)
        elif entry_comparator == '==, Equal To' and exit_comparator == '<, Crossing Down':
            buy_price, sell_price, strategy_signals = equalto_crossingdown(backtestdata, entry_data1, entry_data2,
                                                                           exit_data1, exit_data2)
        elif entry_comparator == '==, Equal To' and exit_comparator == '==, Equal To':
            buy_price, sell_price, strategy_signals = equalto_equalto(backtestdata, entry_data1, entry_data2,
                                                                      exit_data1, exit_data2)

        position = []
        for i in range(len(strategy_signals)):
            if strategy_signals[i] > 1:
                position.append(0)
            else:
                position.append(0)

        for i in range(len(backtestdata.Close)):
            if strategy_signals[i] == 1:
                position[i] = 1
            elif strategy_signals[i] == -1:
                position[i] = 0
            else:
                position[i] = position[i - 1]

        st.caption(f'BACKTEST  RESULTS  FROM  {start_date}  TO  {end_date}')

        st.markdown('')

        buy_hold = backtestdata.Close.pct_change().dropna()
        strategy = (position[1:] * buy_hold).dropna()
        strategy_returns_per = np.exp(strategy.sum()) - 1
        bh_returns_per = np.exp(buy_hold.sum()) - 1

        n_days = len(backtestdata)
        annualized_returns = 252 / n_days * strategy_returns_per

        buy_signals = pd.Series(buy_price).dropna()
        sell_signals = pd.Series(sell_price).dropna()
        total_signals = len(buy_signals) + len(sell_signals)

        max_drawdown = pas.max_dd(strategy)

        profit = []
        losses = []
        for i in range(len(strategy)):
            if strategy[i] > 0:
                profit.append(strategy[i])
            elif strategy[i] < 0:
                losses.append(strategy[i])
            else:
                pass

        profit_factor = pd.Series(profit).sum() / (abs(pd.Series(losses)).sum())

        strat_percentage, bh_percentage, annr = st.columns(3)
        strat_percentage = strat_percentage.metric(label='Strategy Profit Percentage',
                                                   value=f'{round(strategy_returns_per * 100, 2)}%')
        bh_percentage = bh_percentage.metric(label='Buy/Hold Profit Percentage',
                                             value=f'{round(bh_returns_per * 100, 2)}%')
        annr = annr.metric(label='Annualized Return', value=f'{round(annualized_returns * 100, 2)}%')

        nos, md, pf = st.columns(3)
        nos = nos.metric(label='Total No. of Signals', value=f'{total_signals}')
        md = md.metric(label='Max Drawdown', value=f'{round(max_drawdown, 2)}%')
        pf = pf.metric(label='Profit Factor', value=f'{round(profit_factor, 2)}')

        key_visuals = st.expander('KEY VISUALS')

        key_visuals.caption('Strategy Equity Curve')
        scr = pd.DataFrame(strategy.cumsum()).rename(columns={'Close': 'Returns'})
        scr.index = strategy.index
        key_visuals.area_chart(scr)

        key_visuals.markdown('')
        key_visuals.markdown('')


@app.addapp(title='Live Forex Signals')
def liveforex():
    from tradingview_ta import TA_Handler, Interval
    import streamlit as st
    import tradingview_ta, requests, os
    from datetime import timezone
    from st_aggrid import AgGrid
    import pandas as pd
    import yfinance as yf
    from streamlit_echarts import st_echarts
    import matplotlib.pyplot as plt
    exchange = "FX_IDC"
    screener = "forex"

    def get_analysis(symbol):
        # TradingView Technical Analysis
        handler = tradingview_ta.TA_Handler()
        handler.set_symbol_as(symbol)
        handler.set_exchange_as_crypto_or_stock(exchange)
        handler.set_screener_as_stock(screener)
        handler.set_interval_as(interval)
        analysis = handler.get_analysis()

        return analysis

    # Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
    # after it's been refreshed 100 times.
    count = st_autorefresh(interval=120000, limit=100, key="fizzbuzzcounter")

    # The function returns a counter for number of refreshes. This allows the
    # ability to make special requests at different intervals based on the count
    if count == 0:
        st.write("Count is zero")
    elif count % 3 == 0 and count % 5 == 0:
        st.write("FizzBuzz")
    elif count % 3 == 0:
        st.write("Fizz")
    elif count % 5 == 0:
        st.write("Buzz")
    else:
        st.write(f"Count: {count}")

    st.header("Summary Of Indicators")
    interval = st.selectbox("Interval", ("5m", "1m", "15m", "30m", "1h", "4h", "1d", "1W", "1M"))

    st.subheader("Interval is : " + interval)

    col30, col31, col32, col33, col34 = st.columns(5)
    col30.metric("CURRENCY PAIR", "EUR/USD")
    col31.metric("BUY", get_analysis('EURUSD').summary['BUY'])
    col32.metric("SELL", get_analysis('EURUSD').summary['SELL'])
    col33.metric("NEUTRAL", get_analysis('EURUSD').summary['NEUTRAL'])
    col34.metric('RECOMMENDATION', get_analysis('EURUSD').summary['RECOMMENDATION'])

    col30, col31, col32, col33, col34 = st.columns(5)
    col30.metric("CURRENCY PAIR", "EUR/GBP")
    col31.metric("BUY", get_analysis('EURGBP').summary['BUY'])
    col32.metric("SELL", get_analysis('EURGBP').summary['SELL'])
    col33.metric("NEUTRAL", get_analysis('EURGBP').summary['NEUTRAL'])
    col34.metric('RECOMMENDATION', get_analysis('EURGBP').summary['RECOMMENDATION'])

    col30, col31, col32, col33, col34 = st.columns(5)
    col30.metric("CURRENCY PAIR", "USD/JPY")
    col31.metric("BUY", get_analysis('USDJPY').summary['BUY'])
    col32.metric("SELL", get_analysis('USDJPY').summary['SELL'])
    col33.metric("NEUTRAL", get_analysis('USDJPY').summary['NEUTRAL'])
    col34.metric('RECOMMENDATION', get_analysis('USDJPY').summary['RECOMMENDATION'])

    col30, col31, col32, col33, col34 = st.columns(5)
    col30.metric("CURRENCY PAIR", "GBP/USD")
    col31.metric("BUY", get_analysis('GBPUSD').summary['BUY'])
    col32.metric("SELL", get_analysis('GBPUSD').summary['SELL'])
    col33.metric("NEUTRAL", get_analysis('GBPUSD').summary['NEUTRAL'])
    col34.metric('RECOMMENDATION', get_analysis('GBPUSD').summary['RECOMMENDATION'])

    col30, col31, col32, col33, col34 = st.columns(5)
    col30.metric("CURRENCY PAIR", "AUD/USD")
    col31.metric("BUY", get_analysis('AUDUSD').summary['BUY'])
    col32.metric("SELL", get_analysis('AUDUSD').summary['SELL'])
    col33.metric("NEUTRAL", get_analysis('AUDUSD').summary['NEUTRAL'])
    col34.metric('RECOMMENDATION', get_analysis('AUDUSD').summary['RECOMMENDATION'])

    col30, col31, col32, col33, col34 = st.columns(5)
    col30.metric("CURRENCY PAIR", "USD/CAD")
    col31.metric("BUY", get_analysis('USDCAD').summary['BUY'])
    col32.metric("SELL", get_analysis('USDCAD').summary['SELL'])
    col33.metric("NEUTRAL", get_analysis('USDCAD').summary['NEUTRAL'])
    col34.metric('RECOMMENDATION', get_analysis('USDCAD').summary['RECOMMENDATION'])

    col30, col31, col32, col33, col34 = st.columns(5)
    col30.metric("CURRENCY PAIR", "USD/CNY")
    col31.metric("BUY", get_analysis('USDCNY').summary['BUY'])
    col32.metric("SELL", get_analysis('USDCNY').summary['SELL'])
    col33.metric("NEUTRAL", get_analysis('USDCNY').summary['NEUTRAL'])
    col34.metric('RECOMMENDATION', get_analysis('USDCNY').summary['RECOMMENDATION'])

    col30, col31, col32, col33, col34 = st.columns(5)
    col30.metric("CURRENCY PAIR", "USD/CHF")
    col31.metric("BUY", get_analysis('USDCHF').summary['BUY'])
    col32.metric("SELL", get_analysis('USDCHF').summary['SELL'])
    col33.metric("NEUTRAL", get_analysis('USDCHF').summary['NEUTRAL'])
    col34.metric('RECOMMENDATION', get_analysis('USDCHF').summary['RECOMMENDATION'])

    col30, col31, col32, col33, col34 = st.columns(5)
    col30.metric("CURRENCY PAIR", "USD/HKD")
    col31.metric("BUY", get_analysis('USDHKD').summary['BUY'])
    col32.metric("SELL", get_analysis('USDHKD').summary['SELL'])
    col33.metric("NEUTRAL", get_analysis('USDHKD').summary['NEUTRAL'])
    col34.metric('RECOMMENDATION', get_analysis('USDHKD').summary['RECOMMENDATION'])

    col30, col31, col32, col33, col34 = st.columns(5)
    col30.metric("CURRENCY PAIR", "USD/KRW")
    col31.metric("BUY", get_analysis('USDKRW').summary['BUY'])
    col32.metric("SELL", get_analysis('USDKRW').summary['SELL'])
    col33.metric("NEUTRAL", get_analysis('USDKRW').summary['NEUTRAL'])
    col34.metric('RECOMMENDATION', get_analysis('USDKRW').summary['RECOMMENDATION'])







@app.addapp(title='Crypto Prices')
def cryptoprices():
    # !/usr/bin/env python
    # -*- coding: utf-8 -*-
    import streamlit as st
    import pandas as pd
    from modules.data import get_data
    from modules.styles import style_table
    from modules.views import table, buttom
    from modules.funtions import ping



    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

    def box_select_coin(col_position):
        path = '/Users/administrator/PycharmProjects/MultipleApp/financialdashboard/1crypto_viewer_good/helpers/CoinsID.json'
        df = pd.read_json(path)
        box_coins = col_position.multiselect('', df['id'], help='Select coins...')

        if box_coins == []:
            box_coins = None
        else:
            box_coins = str(box_coins).replace('[', '').replace(']', '')

        return box_coins

    if __name__ == '__main__':

        st.title("Cryptocurrency Prices")

        col1, col2, col3 = st.columns(3)
        col1.write('### NumberPage')
        page_number = buttom(col1)
        col3.write('### Select your coin')
        box_coins = box_select_coin(col3)

        df = get_data(page_number, box_coins)

        col2.write('### Select columns view')
        box_cols = col2.multiselect('', df.columns, help='Select columns...')

        if box_cols == []:
            pass
        else:
            box_cols.insert(0, 'LOGO')
            box_cols.insert(1, 'NAME')
            df = df[box_cols]

        if ping() == True:
            df = style_table(df)
            table(df)
        else:
            pass

    # https://github.com/Luisarg03/Streamlit-CryptoCurrency/tree/master/app/modules

@app.addapp(title='Live Stock Signals')
def livvestock():
    hy.info('Hello from Live Stock')

@app.addapp(title='Forex Trading Guide')
def tradeforex():
    st.title("FOREX TRADING GUIDE")
    # after it's been refreshed 100 times.
    count = st_autorefresh(interval=120000, limit=100, key="fizzbuzzcounter")

    # The function returns a counter for number of refreshes. This allows the
    # ability to make special requests at different intervals based on the count
    if count == 0:
        st.write("Count is zero")
    elif count % 3 == 0 and count % 5 == 0:
        st.write("FizzBuzz")
    elif count % 3 == 0:
        st.write("Fizz")
    elif count % 5 == 0:
        st.write("Buzz")
    else:
        st.write(f"Count: {count}")
    interval = st.selectbox("Interval", ("5m", "1m", "15m", "30m", "1h", "4h", "1d", "1W", "1M"))

    symbls = ['USDCAD', 'EURJPY', 'EURUSD', 'EURCHF', 'USDCHF', 'EURGBP', 'GBPUSD', 'AUDCAD', 'NZDUSD',
              'GBPCHF', 'AUDUSD', 'GBPJPY', 'USDJPY', 'CHFJPY', 'EURCAD', 'AUDJPY', 'EURAUD', 'AUDNZD']
    d = {}

    col51, col52, col53, col54 = st.columns(4)

    with col51:
        st.header("STRONG BUY")
        for symbl in symbls:
            output = TA_Handler(
                symbol=symbl,
                screener="forex",
                exchange="FX_IDC",
                interval=interval)
            d[symbl] = output.get_analysis().summary
            for key in d[symbl]:
                if d[symbl][key] == "STRONG_BUY":
                    st.write(symbl)

    with col52:
        st.header("STRONG SELL")
        for symbl in symbls:
            output = TA_Handler(
                symbol=symbl,
                screener="forex",
                exchange="FX_IDC",
                interval=interval)
            d[symbl] = output.get_analysis().summary
            for key in d[symbl]:
                if d[symbl][key] == "STRONG_SELL":
                    st.write(symbl)

    with col53:
        st.header("BUY")
        for symbl in symbls:
            output = TA_Handler(
                symbol=symbl,
                screener="forex",
                exchange="FX_IDC",
                interval=interval)
            d[symbl] = output.get_analysis().summary
            for key in d[symbl]:
                if d[symbl][key] == "BUY":
                    st.write(symbl)

    with col54:
        st.header("SELL")
        for symbl in symbls:
            output = TA_Handler(
                symbol=symbl,
                screener="forex",
                exchange="FX_IDC",
                interval=interval)
            d[symbl] = output.get_analysis().summary
            for key in d[symbl]:
                if d[symbl][key] == "SELL":
                    st.write(symbl)


@app.addapp(title='Crypto Trading Guide')
def tradecrypto():
    hy.info('Hello from Crypto Trading Guide')

@app.addapp(title='Stock Trading Guide')
def tradestock():
    hy.info('Stock Trading Guide')

@app.addapp(title='Crypto Forecaster')
def cryptoforecaster():
    st.title('Crypto Forecaster')
    st.markdown("This application enables you to predict on the future value of any cryptocurrency (available on Coinmarketcap.com), for \
                    	any number of days into the future! The application is built with Streamlit (the front-end) and the Facebook Prophet model, \
                    	which is an advanced open-source forecasting model built by Facebook, running under the hood. You can select to train the model \
                    	on either all available data or a pre-set date range. Finally, you can plot the prediction results on both a normal and log scale.")

    ### Change sidebar color
    st.markdown(
        """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#D6EAF8,#D6EAF8);
        color: black;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    ### Set bigger font style
    st.markdown(
        """
    <style>
    .big-font {
        fontWeight: bold;
        font-size:22px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("Forecaster Settings")

    ### Select ticker & number of days to predict on
    selected_ticker = st.text_input("Select a ticker for prediction (i.e. BTC, ETH, LINK, etc.)", "BTC")
    period = int(
        st.number_input('Number of days to predict:', min_value=0, max_value=1000000, value=365,
                        step=1))
    training_size = int(
        st.number_input('Training set (%) size:', min_value=10, max_value=100, value=100, step=5)) / 100

    ### Initialise scraper without time interval
    @st.cache
    def load_data(selected_ticker):
        init_scraper = CmcScraper(selected_ticker)
        df = init_scraper.get_dataframe()
        min_date = pd.to_datetime(min(df['Date']))
        max_date = pd.to_datetime(max(df['Date']))
        return min_date, max_date

    data_load_state = st.text('Loading data...')
    min_date, max_date = load_data(selected_ticker)
    data_load_state.text('Loading data... done!')

    ### Select date range
    date_range = st.selectbox("Select the timeframe to train the model on:",
                              options=["All available data", "Specific date range"])

    if date_range == "All available data":

        ### Initialise scraper without time interval
        scraper = CmcScraper(selected_ticker)

    elif date_range == "Specific date range":

        ### Initialise scraper with time interval
        start_date = st.date_input('Select start date:', min_value=min_date, max_value=max_date,
                                   value=min_date)
        end_date = st.date_input('Select end date:', min_value=min_date, max_value=max_date,
                                 value=max_date)
        scraper = CmcScraper(selected_ticker, str(start_date.strftime("%d-%m-%Y")),
                             str(end_date.strftime("%d-%m-%Y")))

    ### Pandas dataFrame for the same data
    data = scraper.get_dataframe()

    st.subheader('Raw data')
    st.write(data.head())

    ### Plot functions
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    def plot_raw_data_log():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
        fig.update_yaxes(type="log")
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    ### Plot (log) data
    plot_log = st.checkbox("Plot log scale")
    if plot_log:
        plot_raw_data_log()
    else:
        plot_raw_data()

    ### Predict forecast with Prophet
    if st.button("Predict"):

        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        ### Create Prophet model
        m = Prophet(
            changepoint_range=training_size,  # 0.8
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # multiplicative/additive
            changepoint_prior_scale=0.05
        )

        ### Add (additive) regressor
        for col in df_train.columns:
            if col not in ["ds", "y"]:
                m.add_regressor(col, mode="additive")

        m.fit(df_train)

        ### Predict using the model
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        ### Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.head())

        st.subheader(f'Forecast plot for {period} days')
        fig1 = plot_plotly(m, forecast)
        if plot_log:
            fig1.update_yaxes(type="log")
        st.plotly_chart(fig1)

        st.subheader("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)



@app.addapp(title='Pricing', icon="🥰")
def pricing():
    hy.info('Hello from pricing')

@app.addapp(title='User Guide')
def howtouse():
    hy.info('How to use')

@app.addapp(title='Contact Us')
def contactus():
    hy.info('Hello from contact us')

@app.addapp(title='Crypto Prices 2')
def criptopric():
    ######################################################
    # Stock Technical Analysis with Python               #
    # Average Directional Movement Index ADX(14)         #
    # (c) Diego Fernandez Garcia 2016                    #
    # www.exfinsis.com                                   #
    ######################################################

    # 1. Packages and Data

    # Packages Import
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import datetime
    import pyarrow
    import altair as alt
    import pandas_datareader as web
    import datetime as dt
    import matplotlib.pyplot as plt
    import talib as ta
    # Data Download
    #start = dt.datetime(2014, 10, 1)
    #end = dt.datetime(2015, 9, 30)
    #aapl = web.DataReader('AAPL', 'yahoo', start, end)

    df = yf.download("AAPL", start="2019-01-01", end="2020-01-01", group_by="ticker")
    st.dataframe(df)
    # Technical Indicator Chart
    df['DEMA'] = ta.DEMA(df.Close, 90)
    st.write(df['DEMA'])
    df['ADX'] = ta.ADX(df.High, df.Low, df.Close, 90)
    df['ADX']
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = ta.MACD(df.Close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD']
    df['MACDSIGNAL']
    df['MACDHIST']



    # If the menu is cluttered, just rearrange it into sections!
    # completely optional, but if you have too many entries, you can make it nicer by using accordian menus
if user_access_level > 0:
        complex_nav = {
            'Home': ['Home'],
            'Forex 🏆': ['Forex'],
            'crypto 🔥': ["Crypto Signal", "Crypto Forecaster", "Crypto Prediction", "Crypto Prices 2"],
            'Stock': ["Stock Signals"],
            'Backtesting': ["Backtesting"],
            'Live': ["Live Forex Signals","Crypto Prices", "Live Stock Signals"],
            'What To Trade ?': ["Forex Trading Guide", "Crypto Trading Guide", "Stock Trading Guide"],
            'Pricing': ['Pricing'],
            'How To Use': ['User Guide'],
            'Contactus': ['Contact Us'],
        }
elif user_access_level == 1:
        complex_nav = {
            'Home': ['Home'],
            'Pricing': ['Pricing'],
            'Stock': ["Stock Signals"],
            'How To Use': ['How To Use'],
            'Contactus': ['Contact Us'],
        }
else:
        complex_nav = {
            'Home': ['Home'],
            'pricing': ['Pricing'],
            'How To Use': ['How To Use'],
            'contactus': ['Contact Us'],

        }

    # and finally just the entire app and all the children.


    # (DEBUG) print user movements and current login details used by Hydralit
    # ---------------------------------------------------------------------
user_access_level, username = app.check_access()
prev_app, curr_app = app.get_nav_transition()
st.write(prev_app,'- >', curr_app)
st.write(int(user_access_level),'- >', username)

    # ---------------------------------------------------------------------


app.run(complex_nav)

