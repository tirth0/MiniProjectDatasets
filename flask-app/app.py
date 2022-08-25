from flask import Flask, request, jsonify
import pickle


def detect_text_word(img):
    str_final = ''
    himg, wimg, _ = img.shape
    boxes = pytesseract.image_to_data(img)
    for x, b in enumerate(boxes.splitlines()):
        if x != 0:
            b = b.split()
            if len(b) == 12:
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                cv2.rectangle(img, (x, y), (w+x, h+y), (0, 0, 255), 1)
                cv2.putText(img, b[11], (x, y),
                            cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 1)
                str_final += b[11]
                str_final += ' '
    return img, str_final


def predict_id(path):
    try:
        image = cv2.imread(path)
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.medianBlur(gray, 3)
        ret, thresh = cv2.threshold(gray, 200, 255, 0)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=5)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            crop_img = image[y:y+h, x:x+w]
        img, str_final = detect_text_word(crop_img)
        years_final = []
        years = re.findall(r'\d{4}', str_final)
        for i in years:
            years_final.append(int(i))
        years_final.sort()
        obj = re.search(
            r'[0-9][A-Z][A-Z][0-9][0-9][A-Z][A-Z][0-9][0-9][0-9]', str_final).group()
        return(obj, years_final[-1])
    except:
        return(None, None)


app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify('No file at all ! !')
        file = request.files['file']

        if file.filename == '':
            return jsonify('File Error !')

        if file and allowed_file(file.filename):
            file_path = os.path.join(
                os.getcwd() + UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            usn, last_year = predict_id(file_path)
            if(last_year is not None and date.today().year <= last_year):
                return jsonify({'usn': usn})
            else:
                return jsonify('Invalid registration')
        return jsonify('wrong image')
