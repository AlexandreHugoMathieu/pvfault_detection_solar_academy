pip install virtualenv
python -m virtualenv solar_env
call solar_env\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --name=solarkernel

call solar_env\Scripts\activate
jupyter notebook