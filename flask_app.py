from flask import Flask,render_template,url_for,jsonify,request,redirect
from model import *
from werkzeug.utils import secure_filename
import cv2

# -------App Initialization--------
app=Flask(__name__)
ALLOWED_EXTENSIONS = {'avi','mp4'}

# -------App Definition--------



@app.route("/convlstm")
def convlstm_page():
    return render_template('convlstm.html',
                           title='ConvLSTM',
                           heading='Video Classification using CNN and LSTM')
@app.route("/lrcn")
def conv2_page():
    return render_template('lrcn.html',
                           title='LRCN',
                           heading='Video Classification using CNN and LSTM')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    action1,action2="No Predicted Action","No Predicted Action"
    filename = "No File uploaded"
    conf1,conf2 = "No Predicted Probability","No Predicted Probability"
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            vid = request.files['file']
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vid.read())
            video_reader = cv2.VideoCapture(tfile.name)
            vid1,vid2 = video_reader,video_reader
            # action1,conf1 = predict_single_action_1(vid1)
            action2,conf2 = predict_single_action_2(vid2)
            return render_template('predict.html', conf1=conf1,
                                                   conf2=conf2,
                                                   action1=action1,
                                                   action2=action2,
                                                   filename=filename)
    return render_template('predict.html',
                           conf1=conf1,
                           conf2=conf2,
                           action1=action1,
                           action2=action2,
                           filename=filename)

#--------RUN--------
if __name__ == '__main__':
    app.run(debug=True)