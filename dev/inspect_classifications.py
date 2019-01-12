import json
import flask
import os
import datetime


# flask app to see/inspect classifications
app = flask.Flask(__name__)


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
    return flask.send_from_directory(os.path.join(
        '/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/', _p), _f)


@app.route('/', methods=['GET', 'POST'])
def root():
    classes = {
        0: "Plausible Asteroid (short streak)",
        1: "Satellite (long streak - could be partially masked)",
        2: "Masked bright star",
        3: "Dementors and ghosts",
        4: "Cosmic rays",
        5: "Yin-Yang (multiple badly subtracted stars)",
        6: "Satellite flashes",
        7: "Skip (Includes 'Not Sure' and seemingly 'Blank Images')"
    }

    # TODO: get latest json
    with open('/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/zooniverse.20180822.json', 'r') as f:
        classifications_raw = json.load(f)

    if flask.request.method == 'GET':
        classifications = dict()
        # i = 0
        for crk, crv in classifications_raw.items():
            if os.path.exists(os.path.join('/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/zooniverse',
                                           crk)):
                classifications[crk] = crv
            #     i += 1
            # if i > 4:
            #     break

        return flask.render_template('template-root.html', logo='Zwickyverse',
                                     cutouts=classifications, classes=list(classes.values()))
    elif flask.request.method == 'POST':
        classifications = json.loads(flask.request.get_data())
        classifications = {k: v for k, v in classifications.items() if len(v) > 0}
        date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/zooniverse.{date}.json', 'w') as f:
            json.dump(classifications, f, indent=2)
        return flask.jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(host='localhost', port=3000, threaded=True)