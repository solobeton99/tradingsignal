#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests as rq


def ping():
    '''
    Test connection
    '''
    url = 'https://api.coingecko.com/api/v3/ping'
    if rq.get(url).status_code == 200:
        return True
    else:
        return False