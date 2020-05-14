from http import cookies, cookiejar

class Cookie():
    def __init__(self):
        self.cookie = cookies.SimpleCookie()

c = Cookie()
c.name = 'Aspire Issued'
c.expires = 3600
c.path = '/cookiedomain'
c.comment = ''
c.domain = ''
c.max_age = 60 * 60 * 60
c.secure = True
c.version = ""
c.httponly = True
c.samesite = True

print(dir(c))
