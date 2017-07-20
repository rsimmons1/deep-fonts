import os
from flask import Flask, request, render_template
import model
import numpy
import PIL, PIL.Image, PIL.PngImagePlugin
import StringIO
import jinja2
os.environ['THEANO_FLAGS'] = 'device=cpu'
import theano

model = model.Model(artificial_font=True)
model.try_load()
run_fn = model.get_run_fn()

app = Flask('fonts')

env = jinja2.Environment(loader=jinja2.FileSystemLoader('static'))

@app.route('/')
def root():
    template = env.get_template('index.html')
    return template.render(d=model.d)


@app.route('/font.png')
def font():
    input_i = numpy.array(map(float, flask.request.query_string.split(',')))
    input_i = numpy.array([input_i for z in xrange(62)], dtype=theano.config.floatX)
    input_j = numpy.array(range(62), dtype=numpy.int32)
    pred = run_fn(input_i, input_j).reshape((62, 64, 64))

    img = PIL.Image.new('L', (8 * 64, 8 * 64), 255)
    for z in xrange(62):
        x, y = z % 8, z // 8
        img_char = PIL.Image.fromarray(numpy.uint8(((1.0 - pred[z]) * 255)))
        img.paste(img_char, (x * 64, y * 64))

    s = StringIO.StringIO()
    img.save(s, format='PNG')
    return flask.Response(response=s.getvalue(), mimetype='image/png')

@app.route('/draw')
def draw_route():
    template = env.get_template('draw.html')
    return template.render(d=model.d)

@app.route('/create_font',methods=['POST'])
def create_font():
    import re
    import base64
    dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
    img_data = request.form['img']
    print img_data
    imgb64 = dataUrlPattern.match(img_data).group(2)
    print imgb64
    if imgb64 is not None and len(imgb64) > 0:
        with open("test_img.png","wb") as test_img:
            test_img.write(base64.decodestring(imgb64))
        return 'YAYYY'
    else:
        return 'NOOOO'

def main():
    app.run(host='0.0.0.0', debug=True)

if __name__ == '__main__':
    main()
