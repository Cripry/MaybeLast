FROM node:18.17.1

# Set working directory
WORKDIR /app

# Declare build-time argument and set environment variable
ARG DATABASE_URL
ENV DATABASE_URL=postgresql://myuser:mypassword@db:5432/mydb

# Copy package.json and package-lock.json first
COPY apps/wind-turbine-web/package.json ./
COPY package-lock.json ./

# Install dependencies
RUN npm install
RUN npm install --save-dev @types/styled-components-react-native
RUN npm i --save-dev @types/uuid
RUN npm i date-fns

# Copy the Prisma directory containing the schema.prisma file
COPY apps/wind-turbine-web/prisma ./prisma/
COPY .env   ./


# Install dependencies and build Python 3.9.13 from source
# Install dependencies necessary to fetch and install Python
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    curl

# Download, compile, and install Python 3.9.13
RUN wget https://www.python.org/ftp/python/3.9.13/Python-3.9.13.tgz && \
    tar -xzf Python-3.9.13.tgz && \
    cd Python-3.9.13 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.9.13.tgz Python-3.9.13

# Install pip for Python 3.9
RUN curl -s https://bootstrap.pypa.io/get-pip.py | python3.9

# Install pandas using pip
RUN python3.9 -m pip install pandas
RUN python3.9 -m pip install psycopg2

COPY wait-for-it.sh /app/wait-for-it.sh


# Copy the rest of the application
COPY apps/wind-turbine-web/ .

# Expose the application on port 3000
EXPOSE 3000

