# BUILDING STAGE
FROM maven:3.9-eclipse-temurin-21 AS build

WORKDIR /app

COPY src/pom.xml .
    
RUN mvn dependency:go-offline

COPY src ./src

RUN mvn package -DskipTests

# RUNTIME STAGE

FROM eclipse-temurin:21-jre 

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/target/*jar app.jar

CMD ["java", "-jar", "app.jar"]

EXPOSE 8080

# End of Dockerfile
# This Dockerfile uses a multi-stage build to first compile a Java application using Maven,
# and then create a lightweight runtime image with only the necessary dependencies to run the application.
# It installs additional libraries required for media processing.