import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from io import BytesIO
import clip
import torch

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# تنظیمات پوشه ها
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_UPLOADS'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# تنظیمات دیتابیس
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///locations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# مدل دیتابیس
class Location(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    lat = db.Column(db.Float, nullable=False)
    lon = db.Column(db.Float, nullable=False)
    wiki = db.Column(db.String(500))

# مدل Admin
class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

from werkzeug.security import generate_password_hash, check_password_hash

# اطمینان از وجود پوشه‌ها
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_UPLOADS'], exist_ok=True)

# بارگذاری مدل CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# بررسی فرمت مجاز فایل
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ------------------------------------------------
# روت های اصلی

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded"

    file = request.files['image']
    if not file or not allowed_file(file.filename):
        return "Invalid file type"

    try:
        # پردازش تصویر
        file_bytes = file.read()
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # دریافت همه مکان‌ها از دیتابیس
        locations = Location.query.all()
        labels = [location.name for location in locations]

        if not labels:
            return "No locations available. Please contact admin."

        text_tokens = clip.tokenize(labels).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            similarity = (image_features @ text_features.T).squeeze(0)
            best_match_idx = similarity.argmax().item()
            prediction = labels[best_match_idx]
            predicted_location = locations[best_match_idx]

        # ذخیره تصویر برای نمایش
        filename = file.filename
        save_path = os.path.join(app.config['STATIC_UPLOADS'], filename)
        with open(save_path, 'wb') as f:
            f.write(file_bytes)
            print(f"مسیر ذخیره سازی تصویر: {save_path}")  # چاپ مسیر کامل ذخیره سازی
        print(f"نام فایل: {filename}") # چاپ نام فایل

        return render_template("result.html", filename=filename, location=predicted_location, result=prediction)

    except Exception as e:
        return f"Error: {e}"

# ------------------------------------------------
# مدیریت ادمین

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # نام کاربری و رمز عبور ساده
        if username == 'admin' and password == 'admin':
            session['admin_logged_in'] = True
            flash('Logged in successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('admin/login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    locations = Location.query.all()
    return render_template('admin/dashboard.html', locations=locations)

@app.route('/admin/add', methods=['GET', 'POST'])
def admin_add_location():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        name = request.form['name']
        description = request.form['description']
        lat = float(request.form['lat'])
        lon = float(request.form['lon'])
        wiki = request.form['wiki']

        new_location = Location(name=name, description=description, lat=lat, lon=lon, wiki=wiki)
        db.session.add(new_location)
        db.session.commit()
        flash('Location added successfully!', 'success')
        return redirect(url_for('admin_dashboard'))

    return render_template('admin/add_edit.html', action='Add', location=None)

@app.route('/admin/edit/<int:id>', methods=['GET', 'POST'])
def admin_edit_location(id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    location = Location.query.get_or_404(id)

    if request.method == 'POST':
        location.name = request.form['name']
        location.description = request.form['description']
        location.lat = float(request.form['lat'])
        location.lon = float(request.form['lon'])
        location.wiki = request.form['wiki']

        db.session.commit()
        flash('Location updated successfully!', 'success')
        return redirect(url_for('admin_dashboard'))

    return render_template('admin/add_edit.html', action='Edit', location=location)

@app.route('/admin/delete/<int:id>')
def admin_delete_location(id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    location = Location.query.get_or_404(id)
    db.session.delete(location)
    db.session.commit()
    flash('Location deleted successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

# ------------------------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # برای ساخت دیتابیس در اولین اجرا
        if Admin.query.first() is None: # Create a default admin user if none exists
            default_admin = Admin(username='admin')
            default_admin.set_password('admin')
            db.session.add(default_admin)
            db.session.commit()

    app.run(debug=True, port=5000)