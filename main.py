from flask import Flask, render_template, flash, request,session
#from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import mysql.connector
from werkzeug.utils import secure_filename



import cv2
import numpy as np
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


def to_bin(data):
    """Convert `data` to binary format as string"""
    if isinstance(data, str):
        return ''.join([ format(ord(i), "08b") for i in data ])
    elif isinstance(data, bytes):
        return ''.join([ format(i, "08b") for i in data ])
    elif isinstance(data, np.ndarray):
        return [ format(i, "08b") for i in data ]
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, "08b")
    else:
        raise TypeError("Type not supported.")
def encode(image_name, secret_data):
    # read the image
    image = cv2.imread(image_name)
    # maximum bytes to encode
    n_bytes = image.shape[0] * image.shape[1] * 3 // 8
    print("[*] Maximum bytes to encode:", n_bytes)
    if len(secret_data) > n_bytes:
        raise ValueError("[!] Insufficient bytes, need bigger image or less data.")
    print("[*] Encoding data...")
    # add stopping criteria
    secret_data += "====="
    data_index = 0
    # convert data to binary
    binary_secret_data = to_bin(secret_data)
    # size of data to hide
    data_len = len(binary_secret_data)
    for row in image:
        for pixel in row:
            # convert RGB values to binary format
            r, g, b = to_bin(pixel)
            # modify the least significant bit only if there is still data to store
            if data_index < data_len:
                # least significant red pixel bit
                pixel[0] = int(r[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            if data_index < data_len:
                # least significant green pixel bit
                pixel[1] = int(g[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            if data_index < data_len:
                # least significant blue pixel bit
                pixel[2] = int(b[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            # if data is encoded, just break out of the loop
            if data_index >= data_len:
                break
    return image
def decode(image_name):
    print("[+] Decoding...")
    # read the image
    image = cv2.imread(image_name)
    binary_data = ""
    for row in image:
        for pixel in row:
            r, g, b = to_bin(pixel)
            binary_data += r[-1]
            binary_data += g[-1]
            binary_data += b[-1]
    # split by 8-bits
    all_bytes = [ binary_data[i: i+8] for i in range(0, len(binary_data), 8) ]
    # convert from bits to characters
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "=====":
            break
    return decoded_data[:-5]

#binary error correcting codes python string to binary
# Function to convert a string to its binary representation
def string_to_binary(input_string):
    """
    Converts a string to its binary equivalent.
    Each character is converted to an 8-bit binary representation.
    """
    return ''.join(format(ord(char), '08b') for char in input_string)


# Function to convert binary back to string
def binary_to_string(binary_string):
    """
    Converts a binary string back to its original string.
    Assumes the binary string is a valid sequence of 8-bit ASCII characters.
    """
    n = 8
    return ''.join(chr(int(binary_string[i:i + n], 2)) for i in range(0, len(binary_string), n))


# Function to calculate the parity bits for Hamming(7,4) encoding
def calculate_parity_bits(data_bits):
    """
    Calculate the parity bits for a 4-bit data using Hamming(7,4) encoding.
    """
    if len(data_bits) < 7:
        binary_list = [0] * (7 - len(data_bits)) + data_bits  # Pad with zeros at the beginning

    # Continue with your parity bit calculations...
    # Assuming `binary_list` has at least 7 elements now.
    # Here we use dummy calculations for illustration:
    binary_list[0] = (binary_list[2] + binary_list[4] + binary_list[6]) % 2  # Example P1 calculation
    binary_list[1] = (binary_list[2] + binary_list[5] + binary_list[6]) % 2  # Example P2 calculation
    binary_list[2] = (binary_list[4] + binary_list[5] + binary_list[6]) % 2  # Example P3 calculation

    return binary_list





# Function to detect and correct errors in a received codeword using Hamming(7,4)
def detect_and_correct_error(codeword):
    """
    Detect and correct single-bit errors in a Hamming(7,4) codeword.
    Returns the corrected codeword and the position of the error (if any).
    """
    # Convert codeword to list of integers
    code = [int(bit) for bit in codeword]

    # Calculate the parity bits
    p1 = (code[2] + code[4] + code[6]) % 2
    p2 = (code[2] + code[5] + code[6]) % 2
    p3 = (code[4] + code[5] + code[6]) % 2

    # Check parity
    parity_bits = [p1, p2, p3]

    # Calculate error position
    error_position = p1 * 1 + p2 * 2 + p3 * 4

    if error_position != 0:
        # Flip the bit at the error position (1-based index)
        print(f"Error detected at position: {error_position}")
        code[error_position - 1] = 1 - code[error_position - 1]  # Flip the bit
        print(f"Corrected codeword: {''.join(str(bit) for bit in code)}")
        return ''.join(str(bit) for bit in code)
    else:
        print("No error detected")
        return codeword






@app.route("/")
def homepage():

    return render_template('index.html')
@app.route("/adminhome")
def adminhome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
    cursor = conn.cursor()
    cursor.execute("select * from user")
    data = cursor.fetchall()
    return render_template("adminhome.html", data=data)
@app.route("/userhome")
def userhome():
    uname=session['uname']
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
    cursor = conn.cursor()
    cursor.execute("select * from user where uname='" + uname + "'")
    data = cursor.fetchall()
    return render_template("userhome.html", data=data)

    return render_template('userhome.html')
@app.route("/register")
def register():

    return render_template('register.html')
@app.route("/number")
def number():

    return render_template('number.html')
@app.route("/view1")
def view1():
    uname = session['uname']
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
    cursor = conn.cursor()
    cursor.execute("select * from filetrans where uname='" + uname + "'")
    data = cursor.fetchall()


    return render_template('view1.html',data=data)
@app.route("/imgview")
def imgview():

    uname = session['uname']
    id = request.args.get('id')
    session['did']=id
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
    cursor = conn.cursor()
    cursor.execute("select * from filetrans where uname='" + uname + "'")
    data = cursor.fetchall()


    return render_template('imgview.html',data=data)
@app.route("/amount")
def amount():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM user")
    data = cur.fetchall()

    return render_template('amountdetails.html',data=data)
@app.route("/userview")
def userview():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM user")
    data = cur.fetchall()
    return render_template('userdetails.html', data=data)

@app.route("/admin")
def admin():
    return render_template('AdminLogin.html')
@app.route("/adminlog",methods=['GET','POST'])
def adminlog():
    if request.method == 'POST':
        uname=request.form['uname']
        password=request.form['password']
        print(uname)
        print(password)
        conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
        cursor = conn.cursor()
        cursor.execute("select * from admin where uname='"+uname+"' and password='"+password+"'")
        data=cursor.fetchone()
        if data is None:
            return "user name and password incorrect"
        else:
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
            cursor = conn.cursor()
            cursor.execute("select * from user")
            data = cursor.fetchall()
            return render_template("adminhome.html",data=data)

@app.route("/ukey",methods=['GET','POST'])
def ukey():
    if request.method == 'POST':
        key=request.form['key']
        id=session['did']
        conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
        cursor = conn.cursor()
        cursor.execute("select * from filetrans where id='"+id+"' and key1='"+key+"'")
        data=cursor.fetchone()
        if data is None:
            return "Key incorrect"
        else:

            conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
            cursor = conn.cursor()
            cursor.execute("select * from filetrans where id='"+id+"'")
            data = cursor.fetchall()
            return render_template("view2.html",data=data)

@app.route("/user")
def user():
    return render_template('UserLogin.html')
@app.route("/newregister",methods=['GET','POST'])
def newregister():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        email = request.form['email']
        pnumber = request.form['pnumber']
        address = request.form['address']
        uname = request.form['uname']
        password = request.form['password']



        conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
        cursor = conn.cursor()
        cursor.execute(
            "insert into user values('','" + name + "','" + gender + "','" + address + "','" + email + "','" + pnumber + "','" + uname + "','" + password + "')")
        conn.commit()
        conn.close()
        return render_template("UserLogin.html")

@app.route("/userlog", methods=['GET', 'POST'])
def userlog():
        if request.method == 'POST':
            uname = request.form['uname']
            password = request.form['password']
            session['uname'] = request.form['uname']
            print(uname)
            print(password)
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
            cursor = conn.cursor()
            cursor.execute("select * from user where uname='" + uname + "' and password='" + password + "'")
            data = cursor.fetchone()
            if data is None:
                return "user name and password incorrect"
            else:
                conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
                cursor = conn.cursor()
                cursor.execute("select * from user where uname='" + uname + "' and password='" + password + "'")
                data = cursor.fetchall()
                return render_template("userhome.html",data=data)

@app.route("/view")
def view():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM filetrans")
    data = cur.fetchall()
    return render_template('view.html', data=data)
@app.route("/view3")
def view3():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM amount")
    data = cur.fetchall()
    return render_template('view3.html', data=data)
@app.route("/fileupload",methods=['GET','POST'])
def fileupload():
    if request.method == 'POST':#
        name = request.form['name']
        army_text = request.form['message']
        f = request.files['file']
        f.save("static/uploads/" + secure_filename(f.filename))
        import cv2
        import numpy as np
        from PIL import Image
        def detect_and_crop_head(input_image_path, output_image_path, factor=1.7):
            # Load the pre-trained face detection model from OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Read the input image using PIL
            image = Image.open(input_image_path)

            # Convert PIL image to OpenCV format (BGR)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert the image to grayscale for face detection
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                # Assuming the first face is the target, you can modify this based on your requirements
                x, y, w, h = faces[0]

                # Calculate the new coordinates and dimensions for a 1:1 aspect ratio
                center_x = x + w // 2
                center_y = y + h // 2
                size = int(max(w, h) * factor)
                x_new = max(0, center_x - size // 2)
                y_new = max(0, center_y - size // 2)

                # Crop the head region with a 1:1 aspect ratio
                cropped_head = cv_image[y_new:y_new + size, x_new:x_new + size]
                # Convert the cropped head back to PIL format
                cropped_head_pil = Image.fromarray(cv2.cvtColor(cropped_head, cv2.COLOR_BGR2RGB))
                # Save the cropped head image
                cropped_head_pil.save(output_image_path)
                print("Cropped head saved successfully.")
            else:
                print("No faces detected in the input image.")

        # Example usage

        import secrets
        key1=secrets.token_hex(4)
        conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
        cur = conn.cursor()
        cur.execute("select * from user where uname='"+name+"'")
        data = cur.fetchone()
        #email = data[4]

        #body = "Key---"+key1
        input_image_path = "static/uploads/" + f.filename
        output_image_path = "static/uploads/"+str(key1) +"face.jpg"

        detect_and_crop_head(input_image_path, output_image_path)

        from PIL import Image
        def encode_text_to_image(text, image_path, output_image_path):
            # Convert the text to binary
            binary_text = ''.join(format(ord(char), '08b') for char in text)
            # Open the image
            img = Image.open(image_path)
            img = img.convert('RGB')  # Ensure it's in RGB format
            pixels = img.load()
            # Ensure the image is large enough to store the text
            width, height = img.size
            if len(binary_text) > width * height * 3:  # Each pixel can hold 3 bytes (RGB)
                raise ValueError("The image is too small to hold the given text")

            # Encode the binary text into the image
            idx = 0
            for y in range(height):
                for x in range(width):
                    if idx < len(binary_text):
                        pixel = list(pixels[x, y])  # Get the RGB values of the pixel
                        for i in range(3):  # Modify the R, G, B values
                            if idx < len(binary_text):
                                pixel[i] = pixel[i] & ~1 | int(binary_text[idx])  # Set LSB to binary data
                                idx += 1
                        pixels[x, y] = tuple(pixel)  # Put the modified pixel back into the image

            # Save the image with the encoded text
            img.save(output_image_path)
            print(f"Text successfully encoded into image and saved as {output_image_path}")

        # Example Usage:
        res = ''.join(format(ord(i), '08b') for i in army_text)

        # printing result
        print("The string after binary conversion : " + str(res))
        # Binary data

        def decode_text_from_image(image_path):
            # Open the encoded image
            img = Image.open(image_path)
            img = img.convert('RGB')
            pixels = img.load()

            # Extract the binary data from the image
            binary_text = ''
            width, height = img.size
            for y in range(height):
                for x in range(width):
                    pixel = pixels[x, y]
                    for i in range(3):
                        binary_text += str(pixel[i] & 1)  # Extract the LSB from each color channel

            # Convert the binary data back to text
            decoded_text = ''
            for i in range(0, len(binary_text), 8):
                byte = binary_text[i:i + 8]
                decoded_text += chr(int(byte, 2))
                if decoded_text[-1] == '\x00':  # Stop when we encounter the null character
                    break

            print(f"Decoded Text: {decoded_text}")
            return decoded_text

            # Example Usage:

        decoded_text = decode_text_from_image("static/uploads/" + f.filename)
        print(len(decoded_text))
        if len(decoded_text)>1:
            print("test")
            res2=""
            encode_text_to_image(str(res2), "static/uploads/" + f.filename,
                                 "static/uploads/" + str(key1) + "encoded_image.png")

            res1 = ''.join(format(ord(i), '08b') for i in army_text)



            # printing result
            print("The string after binary conversion : " + str(res1))
            # Example Usage
            # Step 1: Convert the string to binary
            input_string = army_text  # Example string
            binary_string = string_to_binary(input_string)
            print(f"Original String: {input_string}")
            print(f"Binary Representation: {binary_string}")

            # Step 2: Apply Hamming(7,4) encoding for error correction
            encoded_data = calculate_parity_bits(
                [int(bit) for bit in binary_string[:4]])  # Encode the first 4 bits of the binary string
            print(f"Encoded data with parity bits: {encoded_data}")

            # Step 3: Simulate an error by flipping a random bit (for demonstration)
            error_encoded_data = encoded_data[:]
            error_encoded_data = error_encoded_data[:4] + [0] + error_encoded_data[
                                                                5:]  # Simulate an error at bit position 5
            print(f"Encoded data with error: {error_encoded_data}")

            # Step 4: Detect and correct errors in the received codeword
            corrected_data = detect_and_correct_error(error_encoded_data)

            # Step 5: Convert the corrected binary back to the string
            # We first need to remove the parity bits before converting back to string (extracting only the original data)
            corrected_data_without_parity = corrected_data[2] + corrected_data[4] + corrected_data[5] + corrected_data[
                6]
            decoded_string = binary_to_string(corrected_data_without_parity)
            print(f"Decoded String after Error Correction: {decoded_string}")
            res1 = str(res1) + str(',')
            encode_text_to_image(str(res1), "static/uploads/" + f.filename,"static/uploads/" + str(key1) + "encoded_image.png")

        else:
            encode_text_to_image(str(res), "static/uploads/" + f.filename,
                                 "static/uploads/" + str(key1) + "encoded_image.png")

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
        cursor = conn.cursor()
        key2 = secrets.token_hex(4)
        cursor.execute(
            "insert into filetrans values('','" + name + "','" + f.filename + "','" + army_text + "','"+key1+"','"+key2+"','"+output_image_path+"','"+str(res)+"')")
        conn.commit()
        conn.close()
        return render_template("result1.html",data1=res,data2=army_text,data3="static/uploads/" + f.filename,data4=output_image_path)


@app.route("/verimg",methods=['GET','POST'])
def verimg():
    if request.method == 'POST':#


        f = request.files['file']
        f.save("static/uploads/" + secure_filename(f.filename))
        import cv2
        import numpy as np
        from PIL import Image

        def detect_and_crop_head(input_image_path, output_image_path, factor=1.7):
            # Load the pre-trained face detection model from OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Read the input image using PIL
            image = Image.open(input_image_path)

            # Convert PIL image to OpenCV format (BGR)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert the image to grayscale for face detection
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                # Assuming the first face is the target, you can modify this based on your requirements
                x, y, w, h = faces[0]

                # Calculate the new coordinates and dimensions for a 1:1 aspect ratio
                center_x = x + w // 2
                center_y = y + h // 2
                size = int(max(w, h) * factor)
                x_new = max(0, center_x - size // 2)
                y_new = max(0, center_y - size // 2)

                # Crop the head region with a 1:1 aspect ratio
                cropped_head = cv_image[y_new:y_new + size, x_new:x_new + size]

                # Convert the cropped head back to PIL format
                cropped_head_pil = Image.fromarray(cv2.cvtColor(cropped_head, cv2.COLOR_BGR2RGB))

                # Save the cropped head image
                cropped_head_pil.save(output_image_path)
                print("Cropped head saved successfully.")
            else:
                print("No faces detected in the input image.")

        # Example usage
        input_image_path = "static/uploads/" + f.filename
        output_image_path = "static/uploads/testcropped_head.jpg"

        detect_and_crop_head(input_image_path, output_image_path)


        from PIL import Image


        # Example Usage:

        def decode_text_from_image(image_path):
            # Open the encoded image
            img = Image.open(image_path)
            img = img.convert('RGB')
            pixels = img.load()

            # Extract the binary data from the image
            binary_text = ''
            width, height = img.size
            for y in range(height):
                for x in range(width):
                    pixel = pixels[x, y]
                    for i in range(3):
                        binary_text += str(pixel[i] & 1)  # Extract the LSB from each color channel

            # Convert the binary data back to text
            decoded_text = ''
            for i in range(0, len(binary_text), 8):
                byte = binary_text[i:i + 8]
                decoded_text += chr(int(byte, 2))
                if decoded_text[-1] == '\x00':  # Stop when we encounter the null character
                    break

            print(f"Decoded Text: {decoded_text}")
            return decoded_text


        # Example Usage:
        decoded_text = decode_text_from_image("static/uploads/" + f.filename)
        print(decoded_text)
        print(len(decoded_text))
        txt = decoded_text

        x = txt.split(',')

        print(x[0])

        def is_binary_string(s):
            # use set comprehension to extract all unique characters from the string
            unique_chars = {c for c in s}
            # check if the unique characters are only 0 and 1
            return unique_chars.issubset({'0', '1'})

        def is_binary_string(s):
            # use set comprehension to extract all unique characters from the string
            unique_chars = {c for c in s}
            # check if the unique characters are only 0 and 1
            return unique_chars.issubset({'0', '1'})

        s=is_binary_string(x[0])
        print(s)
        if s==True:
            print(s)
            b = x[0]  # binary for 'geek'

            corrected_data = detect_and_correct_error(x[0])

            # Step 5: Convert the corrected binary back to the string
            # We first need to remove the parity bits before converting back to string (extracting only the original data)
            corrected_data_without_parity = corrected_data[2] + corrected_data[4] + corrected_data[5] + corrected_data[
                6]
            decoded_string = binary_to_string(corrected_data_without_parity)
            print(f"Decoded String after Error Correction: {decoded_string}")

            # Split the binary string into chunks of 8 bits (1 byte)
            n = [b[i:i + 8] for i in range(0, len(b), 8)]

            # Convert binary to string
            s = ''.join(chr(int(i, 2)) for i in n)
            print(s)

            conn = mysql.connector.connect(user='root', password='', host='localhost', database='faceMDT')
            # cursor = conn.cursor()
            cur = conn.cursor()
            cur.execute("SELECT * FROM filetrans where bstr='" + str(x[0])+ "'")
            data = cur.fetchall()
            if data is None:
                status = "Fake"
                print(status)
            else:
                status = "Real"
                print(status)
            return render_template("result.html", data1=status, data2='',
                                   data3="static/uploads/" + f.filename, data4="static/uploads/testcropped_head.jpg")
        else:
            status="Fake Image No Binary code Detected"

            return render_template("result.html", data1=status, data2=decoded_text,
                                   data3="static/uploads/" + f.filename, data4="static/uploads/testcropped_head.jpg")


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
