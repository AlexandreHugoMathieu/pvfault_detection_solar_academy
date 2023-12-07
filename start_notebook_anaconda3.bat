C:\anaconda3\python -m pip install virtualenv
C:\anaconda3\python -m virtualenv solar_env
call solar_env\Scripts\activate
C:\anaconda3\python -m pip install -r requirements.txt
C:\anaconda3\python -m ipykernel install --name=solarkernel

call solar_env\Scripts\activate
jupyter notebook