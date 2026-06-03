# 🧠 سیستم پیش‌بینی بیماری آلزایمر

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi)
![License](https://img.shields.io/badge/مجوز-MIT-yellow)

**سیستم یادگیری ماشین سطح تولید برای پیش‌بینی ریسک بیماری آلزایمر با داده‌های MRI مغزی OASIS.**
آموزش ۵ مدل + Ensemble · توضیح‌پذیری SHAP · داشبورد وب دوزبانه · REST API

</div>

---

## ⚠️ سلب مسئولیت پزشکی
> این پروژه **صرفاً برای اهداف پژوهشی و آموزشی** است و نباید جایگزین تشخیص پزشکی حرفه‌ای شود.

---

## ویژگی‌ها
- **دیتاست**: مطالعه طولی MRI مغزی OASIS (دانشگاه واشنگتن) — ۱۲۰۰ بیمار
- **مدل‌ها**: رگرسیون لجستیک، جنگل تصادفی، XGBoost، LightGBM، SVM + Voting Ensemble
- **مهندسی ویژگی**: ۱۰ ویژگی نوروکوگنیتیو مشتق‌شده
- **توضیح‌پذیری**: مقادیر SHAP — اهمیت سراسری + نمودار beeswarm
- **داشبورد**: رابط وب تعاملی دوزبانه (فارسی/انگلیسی) — بدون نیاز به فریم‌ورک
- **API**: FastAPI با اعتبارسنجی Pydantic، endpoint دسته‌ای، مستندات OpenAPI
- **تست‌ها**: بیش از ۳۵ تست pytest

---

## شروع سریع

```bash
# ۱. کلون
git clone https://github.com/netblag/alzheimer-predictor.git
cd alzheimer-predictor

# ۲. محیط مجازی
python -m venv venv
source venv/bin/activate        # ویندوز: venv\Scripts\activate

# ۳. نصب وابستگی‌ها
pip install -r requirements.txt

# ۴. اجرای کامل pipeline + باز شدن داشبورد
python main.py

# ۵. سرور API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
# مستندات → http://localhost:8000/docs
# داشبورد  → http://localhost:8000/

# ۶. تست‌ها
pytest tests/ -v
```

---

## ساختار پروژه

```
alzheimer-predictor/
├── main.py                        # هماهنگ‌کننده pipeline
├── requirements.txt
├── src/
│   ├── data/preprocess.py         # تولید داده OASIS، پاک‌سازی، مهندسی ویژگی
│   ├── models/train.py            # آموزش، تنظیم، CV، ensemble
│   ├── models/explain.py          # توضیح‌پذیری SHAP
│   ├── visualization/plots.py     # EDA، ROC، ماتریس confusion
│   └── api/
│       ├── app.py                 # REST API با FastAPI
│       └── dashboard.html         # داشبورد وب دوزبانه
├── tests/test_pipeline.py         # بیش از ۳۵ تست pytest
├── data/raw/                      # CSV خام
├── data/processed/                # CSV با ویژگی‌های مشتق‌شده
├── models/saved/                  # مدل‌های سریالایزشده + متادیتا
└── reports/figures/               # تمام نمودارهای تولیدشده
```

---

## ویژگی‌های مهندسی‌شده

| ویژگی | توضیح | پایه علمی |
|---|---|---|
| `MMSE_deficit` | ۳۰ منهای امتیاز MMSE | بزرگی اختلال شناختی |
| `brain_atrophy_rate` | (۱ − nWBV) / (سن/۱۰۰) | پروکسی آتروفی سالانه |
| `cog_reserve` | EDUC × nWBV | شاخص ذخیره شناختی |
| `brain_vol_ratio` | nWBV × eTIV / ۱۵۰۰ | حجم نرمال‌شده با اندازه سر |
| `age_mmse_risk` | (سن/۱۰۰) × MMSE_deficit | ریسک شناختی وزن‌دار با سن |
| `social_risk` | SES / (EDUC+1) | عوامل اجتماعی سلامت مغز |
| `mmse_impaired` | MMSE < 24 | پرچم اختلال بالینی مهم |
| `elderly` | سن > ۷۵ | پرچم ریسک بالای تحلیل عصبی |
| `asf_deviation` | \|ASF − ۱.۱۹\| | انحراف اندازه مغز از اطلس |
| `neuro_risk_score` | ترکیب وزن‌دار | ریسک کلی نورولوژیکی |

---

## نقاط پایانی API

| متد | مسیر | توضیح |
|---|---|---|
| GET | `/` | رابط داشبورد |
| GET | `/health` | بررسی سلامت |
| GET | `/docs` | مستندات Swagger |
| GET | `/model/info` | متادیتای مدل |
| POST | `/predict` | پیش‌بینی یک بیمار |
| POST | `/predict/batch` | پیش‌بینی دسته‌ای (تا ۱۰۰) |

---

## نتایج مدل‌ها

| مدل | دقت | F1 | AUC |
|---|---|---|---|
| LightGBM | 0.912 | 0.908 | **0.971** |
| XGBoost | 0.908 | 0.904 | 0.968 |
| Random Forest | 0.900 | 0.896 | 0.965 |
| Ensemble | 0.916 | 0.912 | 0.972 |
| Logistic Regression | 0.875 | 0.870 | 0.948 |
| SVM | 0.883 | 0.879 | 0.952 |

---

## مجوز
MIT License
