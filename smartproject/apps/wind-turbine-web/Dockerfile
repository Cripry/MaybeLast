FROM node:18.17.1

# Set working directory
WORKDIR /app

# Declare build-time argument and set environment variable
ARG DATABASE_URL
ENV DATABASE_URL=postgresql://myuser:mypassword@db2:5432/mydb

# Copy package.json and package-lock.json first
COPY apps/wind-turbine-web/package.json ./
COPY package-lock.json ./

# Install dependencies
RUN npm install
RUN npm install --save-dev @types/styled-components-react-native

# Copy the Prisma directory containing the schema.prisma file
COPY apps/wind-turbine-web/prisma ./prisma/
COPY .env   ./

# Run Prisma migrations
RUN npx prisma migrate dev --name first-migration

# Copy the rest of the application
COPY apps/wind-turbine-web/ .

# Build the Next.js app
RUN npm run build

# Expose the application on port 3000
EXPOSE 3000

# Start the Next.js app
CMD ["npm", "start"]
