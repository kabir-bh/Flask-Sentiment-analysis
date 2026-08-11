import datetime

DEBUG = False
TESTING = False

SECRET_KEY = 'dev-secret-key-for-local-testing'
CSRF_ENABLED = True

# site
SITE_URL = '/'
SITE_ABOUT = '/about'
SITE_AUTHOR = '/Jesse Zhen'
SITE_TITLE = 'Sentiment Analysis'
SITE_TIME = datetime.datetime.today()
