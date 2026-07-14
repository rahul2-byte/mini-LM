# Local single-node MinIO object storage for development and evaluation.
# Pin this tag before using the image in a reproducible deployment.
FROM minio/minio:latest

EXPOSE 9000 9001

VOLUME ["/data"]

ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
