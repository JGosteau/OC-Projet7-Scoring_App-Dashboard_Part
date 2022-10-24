from dashboard.app import app
server = app.server

if __name__ == '__main__' :
    server.run(port = 8050, debug = True)