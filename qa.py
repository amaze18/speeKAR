"""Speekar chatbot.ipynb



Original file is located at
    https://colab.research.google.com/drive/1GYmsZSR4MWuvORNpSWFWrXz79lQKb6oc
"""

import os

RESULTS_DIR = "scraped_files/"
os.makedirs(RESULTS_DIR, exist_ok=True)
import streamlit as st
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import numpy as np

