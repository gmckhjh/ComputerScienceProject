#
FROM python:3.8

#
WORKDIR /Api

#
COPY ./requirements.txt /Api/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /Api/requirements.txt

#
COPY ./api /Api/api

#
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]
