import inspect
import json
import shutil
import io
import zipfile
from collections import ChainMap
import pymongo

import flask
import flask_login
import flask_pymongo
from flask_jwt_extended import JWTManager, jwt_required, jwt_optional, create_access_token, get_jwt_identity
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash
from bson.json_util import loads, dumps
from bson import ObjectId
import datetime
import pytz
import logging
from ast import literal_eval
import requests
import numpy as np


def get_config(_config_file='/app/config.json'):
    """
        load config data in json format
    """
    try:
        ''' script absolute location '''
        abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

        if _config_file[0] not in ('/', '~'):
            if os.path.isfile(os.path.join(abs_path, _config_file)):
                config_path = os.path.join(abs_path, _config_file)
            else:
                raise IOError('Failed to find config file')
        else:
            if os.path.isfile(_config_file):
                config_path = _config_file
            else:
                raise IOError('Failed to find config file')

        with open(config_path) as cjson:
            config_data = json.load(cjson)
            # config must not be empty:
            if len(config_data) > 0:
                return config_data
            else:
                raise Exception('Failed to load config file')

    except Exception as _e:
        print(_e)
        raise Exception('Failed to read in the config file')


def utc_now():
    return datetime.datetime.now(pytz.utc)


def to_pretty_json(value):
    # return dumps(value, indent=4)  # , separators=(',', ': ')
    return dumps(value, separators=(',', ': '))


def init_db():
    _client = pymongo.MongoClient(username=config['database']['admin'],
                                  password=config['database']['admin_pwd'],
                                  host=config['database']['host'],
                                  port=config['database']['port'])
    # _id: db_name.user_name
    user_ids = [_u['_id'] for _u in _client.admin.system.users.find({}, {'_id': 1})]

    db_name = config['database']['db']
    username = config['database']['user']

    # print(f'{db_name}.{username}')
    # print(user_ids)

    if f'{db_name}.{username}' not in user_ids:
        _client[db_name].command('createUser', config['database']['user'],
                                 pwd=config['database']['pwd'], roles=['readWrite'])
        print('Successfully initialized db')


def add_admin():
    """
        Create admin user for the web interface if it does not exists already
    :param _mongo:
    :param _secrets:
    :return:
    """
    ex_admin = mongo.db.users.find_one({'_id': secrets['database']['admin_username']})
    if ex_admin is None or len(ex_admin) == 0:
        try:
            mongo.db.users.insert_one({'_id': secrets['database']['admin_username'],
                                       'password': generate_password_hash(secrets['database']['admin_password']),
                                       'permissions': {},
                                       'last_modified': utc_now()
                                       })
        except Exception as e:
            print(e)


''' load config '''
config = get_config('/app/config.json')

''' load secrets '''
with open('/app/secrets.json') as sjson:
    secrets = json.load(sjson)

''' initialize the Flask app '''
app = flask.Flask(__name__)
# add 'do' statement to jinja environment (does the same as {{ }}, but returns nothing):
app.jinja_env.add_extension('jinja2.ext.do')

# add json prettyfier
app.jinja_env.filters['tojson_pretty'] = to_pretty_json

# set up secret keys:
app.secret_key = config['server']['SECRET_KEY']
app.config['JWT_SECRET_KEY'] = config['server']['SECRET_KEY']

# config db for regular use
app.config["MONGO_URI"] = f"mongodb://{config['database']['user']}:{config['database']['pwd']}@" + \
                          f"{config['database']['host']}:{config['database']['port']}/{config['database']['db']}"
mongo = flask_pymongo.PyMongo(app)

# Setup the Flask-JWT-Extended extension
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(days=30)
jwt = JWTManager(app)

# session lifetime for registered users
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=365)

# init admin:
init_db()

# add admin if run first time:
add_admin()

''' login management'''
login_manager = flask_login.LoginManager()
login_manager.init_app(app)


class User(flask_login.UserMixin):
    pass


@login_manager.user_loader
def user_loader(username):
    select = mongo.db.users.find_one({'_id': username})
    if select is None:
        # return None
        return

    user = User()
    user.id = username
    return user


@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')
    # look up in the database
    select = mongo.db.users.find_one({'_id': username})
    if select is None:
        return

    user = User()
    user.id = username

    try:
        user.is_authenticated = check_password_hash(select['password'], flask.request.form['password'])

    except Exception as _e:
        print(_e)
        # return None
        return

    return user


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
        Endpoint for login through the web interface
    :return:
    """
    # print(flask_login.current_user)
    if flask.request.method == 'GET':
        # logged in already?
        if flask_login.current_user.is_authenticated:
            return flask.redirect(flask.url_for('root'))
        # serve template if not:
        else:
            return flask.render_template('template-login.html', logo=config['server']['logo'])
    # print(flask.request.form['username'], flask.request.form['password'])

    # print(flask.request)

    username = flask.request.form['username']
    password = flask.request.form['password']
    # check if username exists and passwords match
    # look up in the database first:
    select = mongo.db.users.find_one({'_id': username})
    if select is not None and check_password_hash(select['password'], password):
        user = User()
        user.id = username

        # get a JWT token to use API:
        try:
            # post username and password, get access token
            auth = requests.post('http://localhost:{}/auth'.format(config['server']['port']),
                                 json={"username": username, "password": password})
            access_token = auth.json()['access_token'] if 'access_token' in auth.json() else 'FAIL'
        except Exception as e:
            print(e)
            access_token = 'FAIL'

        user.access_token = access_token
        # print(user, user.id, user.access_token)
        # save to session:
        flask.session.permanent = True
        flask.session['access_token'] = access_token

        flask_login.login_user(user, remember=True)
        return flask.redirect(flask.url_for('root'))
    else:
        # serve template with flag fail=True to display fail message
        return flask.render_template('template-login.html', logo=config['server']['logo'],
                                     messages=[(u'Failed to log in.', u'danger')])


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    """
        Log user out
    :return:
    """
    if 'access_token' in flask.session:
        flask.session.pop('access_token')
        flask.session.modified = True

    flask_login.logout_user()
    return flask.redirect(flask.url_for('root'))


@app.errorhandler(500)
def internal_error(error):
    return '500 error'


@app.errorhandler(404)
def not_found(error):
    return '404 error'


@app.errorhandler(403)
def forbidden(error):
    return '403 error: forbidden'


@login_manager.unauthorized_handler
def unauthorized_handler():
    return flask.redirect(flask.url_for('login'))


# manage users
@app.route('/users', methods=['GET'])
@flask_login.login_required
def manage_users():
    if flask_login.current_user.id == secrets['database']['admin_username']:
        # fetch users from the database:
        _users = {}

        cursor = mongo.db.users.find()
        for usr in cursor:
            # print(usr)
            _users[usr['_id']] = {'permissions': {}}
            for project in usr['permissions']:
                _users[usr['_id']]['permissions'][project] = {}
                _users[usr['_id']]['permissions'][project]['role'] = usr['permissions'][project]['role']
                # _users[usr['_id']]['permissions'][project]['classifications'] = 'NOT DISPLAYED HERE'
        cursor.close()

        return flask.render_template('template-users.html',
                                     user=flask_login.current_user.id,
                                     logo=config['server']['logo'],
                                     users=_users,
                                     current_year=datetime.datetime.now().year)
    else:
        flask.abort(403)


@app.route('/users', methods=['PUT'])
@flask_login.login_required
def add_user():
    """
        Add new user to DB
    :return:
    """
    if flask_login.current_user.id == secrets['database']['admin_username']:
        try:
            username = flask.request.json.get('user', None)
            password = flask.request.json.get('password', None)
            permissions = flask.request.json.get('permissions', '{}')

            if len(username) == 0 or len(password) == 0:
                return 'username and password must be set'

            if len(permissions) == 0:
                permissions = '{}'

            # add user to coll_usr collection:
            mongo.db.users.insert_one(
                {'_id': username,
                 'password': generate_password_hash(password),
                 'permissions': literal_eval(str(permissions)),
                 'last_modified': datetime.datetime.now()}
            )

            return 'success'

        except Exception as _e:
            print(_e)
            return str(_e)
    else:
        flask.abort(403)


@app.route('/users', methods=['POST'])
@flask_login.login_required
def edit_user():
    """
        Edit user info
    :return:
    """

    if flask_login.current_user.id == secrets['database']['admin_username']:
        try:
            _id = flask.request.json.get('_user', None)
            username = flask.request.json.get('edit-user', '')
            password = flask.request.json.get('edit-password', '')
            # permissions = flask.request.json.get('edit-permissions', '{}')

            if _id == secrets['database']['admin_username'] and username != secrets['database']['admin_username']:
                return 'Cannot change the admin username!'

            if len(username) == 0:
                return 'username must be set'

            # change username:
            if _id != username:
                select = mongo.db.users.find_one({'_id': _id})
                select['_id'] = username
                mongo.db.users.insert_one(select)
                mongo.db.users.delete_one({'_id': _id})

            # change password:
            if len(password) != 0:
                result = mongo.db.users.update(
                    {'_id': username},
                    {
                        '$set': {
                            'password': generate_password_hash(password)
                        },
                        '$currentDate': {'last_modified': True}
                    }
                )

            # change permissions:
            # if len(permissions) != 0:
            #     select = mongo.db.users.find_one({'_id': username}, {'_id': 0, 'permissions': 1})
            #     # print(select)
            #     # print(permissions)
            #     _p = literal_eval(str(permissions))
            #     # print(_p)
            #     if str(permissions) != str(select['permissions']):
            #         result = mongo.db.users.update(
            #             {'_id': _id},
            #             {
            #                 '$set': {
            #                     'permissions': _p
            #                 },
            #                 '$currentDate': {'last_modified': True}
            #             }
            #         )

            return 'success'
        except Exception as _e:
            print(_e)
            return str(_e)
    else:
        flask.abort(403)


@app.route('/users', methods=['DELETE'])
@flask_login.login_required
def remove_user():
    """
        Remove user from DB
    :return:
    """
    if flask_login.current_user.id == secrets['database']['admin_username']:
        try:
            # get username from request
            username = flask.request.json.get('user', None)
            if username == secrets['database']['admin_username']:
                return 'Cannot remove the superuser!'

            # try to remove the user:
            mongo.db.users.delete_one({'_id': username})

            return 'success'
        except Exception as _e:
            print(_e)
            return str(_e)
    else:
        flask.abort(403)


@app.route('/auth', methods=['POST'])
def auth():
    """
        Issue a JSON web token (JWT) for a registered user.
        To be used with API
    :return:
    """
    try:
        if not flask.request.is_json:
            return flask.jsonify({"msg": "Missing JSON in request"}), 400

        username = flask.request.json.get('username', None)
        password = flask.request.json.get('password', None)
        if not username:
            return flask.jsonify({"msg": "Missing username parameter"}), 400
        if not password:
            return flask.jsonify({"msg": "Missing password parameter"}), 400

        # check if username exists and passwords match
        # look up in the database first:
        select = mongo.db.users.find_one({'_id': username})
        if select is not None and check_password_hash(select['password'], password):
            # Identity can be any data that is json serializable
            access_token = create_access_token(identity=username)
            return flask.jsonify(access_token=access_token), 200
        else:
            return flask.jsonify({"msg": "Bad username or password"}), 401

    except Exception as _e:
        print(_e)
        return flask.jsonify({"msg": "Something unknown went wrong"}), 400


@app.route('/data/<path:filename>')
# @flask_login.login_required
def data_static(filename):
    """
        Get files
    :param filename:
    :return:
    """
    _p, _f = os.path.split(filename)
    print(_p, _f)
    return flask.send_from_directory(os.path.join(config['path']['path_data'], _p), _f)


def stream_template(template_name, **context):
    """
        see: http://flask.pocoo.org/docs/0.11/patterns/streaming/
    :param template_name:
    :param context:
    :return:
    """
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.enable_buffering(5)
    return rv


@app.route('/', methods=['GET', 'POST'])
@flask_login.login_required
def root():
    user_id = str(flask_login.current_user.id)
    if 'access_token' in flask.session:
        access_token = flask.session['access_token']
    else:
        access_token = None
        return flask.redirect(flask.url_for('login'))

    form = ''
    data = []
    messages = []

    # if flask.request.method == 'GET':
    #     pass

    if flask.request.method == 'POST':
        try:
            form = flask.request.form
            # print(form)
            query = dict()
            query['filter'] = literal_eval(form['filter'])
            # FIXME?
            query['limit'] = int(form['limit']) if 'limit' in form else None

            # FIXME:
            # query['projection'] = {'ades': 0}
            query['projection'] = {}

            # query own API:
            if access_token is not None:
                r = requests.post(os.path.join('http://', f"localhost:{config['server']['port']}", 'streaks'),
                                  json=query,
                                  headers={'Authorization': 'Bearer {:s}'.format(access_token)})
            else:
                r = requests.post(os.path.join('http://', f"localhost:{config['server']['port']}", 'streaks'),
                                  json=query)

            _data = r.json()
            # print(_data)

            if len(_data) == 0:
                messages = [(u'Did not find anything.', u'info')]

            data = _data

        except Exception as e:
            print(e)
            messages = [(u'Failed to digest query.', u'danger')]

    return flask.render_template('template-root.html',
                                 logo=config['server']['logo'],
                                 user=user_id,
                                 form=form,
                                 data=data,
                                 messages=messages)


''' API '''


@app.route('/streaks', strict_slashes=False, methods=['POST'])
# @jwt_optional
@jwt_required
def streaks():
    try:
        try:
            current_user = get_jwt_identity()

            # print(current_user)

            if current_user is not None:
                user_id = str(current_user)

            else:
                # unauthorized
                # return flask.jsonify({"msg": "Unauthorized access attempt"}), 401
                user_id = None

        except Exception as e:
            print(e)
            user_id = None

        query = flask.request.json
        # print(query)

        # prevent fraud: TODO: can add custom user permissions in the future
        if user_id is None:
            # query['filter'] = {'$and': [{'candidate.programid': 1}, query['filter']]}
            flask.abort(403)

        if len(query['projection']) == 0:
            if query['limit'] is None:
                cursor = mongo.db[config['database']['db']].find(query['filter'])  # .limit(2)
            else:
                cursor = mongo.db[config['database']['db']].find(query['filter']).limit(query['limit'])
        else:
            if query['limit'] is None:
                cursor = mongo.db[config['database']['db']].find(query['filter'], query['projection'])  # .limit(2)
            else:
                cursor = mongo.db[config['database']['db']].find(query['filter'],
                                                                 query['projection']).limit(query['limit'])

        _data = list(cursor) if cursor is not None else []

        return flask.Response(dumps(_data), mimetype='application/json')

    except Exception as _e:
        # FIXME: this is for debugging
        print(_e)
        return str(_e)


if __name__ == '__main__':
    app.run(host=config['server']['host'], port=config['server']['port'], threaded=True)
