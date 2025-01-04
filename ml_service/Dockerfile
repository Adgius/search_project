FROM python:3.11.6

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN python -m nltk.downloader punkt

COPY . .

CMD [ "python", "./main.py" ]
