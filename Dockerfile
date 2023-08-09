FROM redhat/ubi9

WORKDIR /app

RUN dnf install -y python3 python3-pip
RUN yum install -y java-11-openjdk

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python3", "stackbraf.py"]
