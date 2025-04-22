from setuptools import setup, find_packages

setup(
    name='aman',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "joblib==1.1.0",
        "tensorflow-cpu==2.5.0",
        "flask==2.0.3",
        "flask-cors==3.0.10",
        "flask-swagger-ui==3.36.0"
    ],
    entry_points={
        'console_scripts': [
            'aman-train_model1=src.train_model1:main',
            'aman-train_model2=src.train_model2:main',
            'aman-predict_model1=src.predict_model1:main',
            'aman-predict_model2=src.predict_model2:main',
            'aman-evaluate_model1=src.evaluate_model1:main',
            'aman-evaluate_model2=src.evaluate_model2:main',
        ]
    }
)