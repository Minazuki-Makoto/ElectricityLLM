from io import BytesIO
from pathlib import Path

from minio import Minio
from minio.error import S3Error


class MinioService:
    def __init__(
            self,
            minio_endpoint,
            minio_access_key,
            minio_secret_key,
            minio_bucket_name,
            secure=False
    ):
        self.bucket = minio_bucket_name
        self.minio_client = Minio(
            minio_endpoint,
            minio_access_key,
            minio_secret_key,
            secure=secure
        )
        self.initialize_bucket()

    def initialize_bucket(self, bucket_name=None):
        bucket_name = bucket_name or self.bucket

        if self.minio_client.bucket_exists(bucket_name):
            print(f"Bucket already exists: {bucket_name}")
        else:
            self.minio_client.make_bucket(bucket_name)
            print(f"Bucket created: {bucket_name}")

        return bucket_name

    def upload_file(
            self,
            object_name,
            data,
            content_type="application/octet-stream",
            bucket_name=None,
            length=None
    ):
        """Upload bytes, a local path, or a seekable binary stream."""
        bucket_name = bucket_name or self.bucket
        close_stream = False

        if isinstance(data, (bytes, bytearray)):
            stream = BytesIO(data)
            length = len(data)
        elif isinstance(data, (str, Path)):
            file_path = Path(data)
            stream = file_path.open("rb")
            length = file_path.stat().st_size
            close_stream = True
        elif hasattr(data, "read"):
            stream = data
            if length is None:
                try:
                    current_position = stream.tell()
                    stream.seek(0, 2)
                    length = stream.tell() - current_position
                    stream.seek(current_position)
                except (AttributeError, OSError) as error:
                    raise ValueError(
                        "length is required for a non-seekable stream"
                    ) from error
        else:
            raise TypeError(
                "data must be bytes, a local path, or a binary stream"
            )

        try:
            return self.minio_client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=stream,
                length=length,
                content_type=content_type
            )
        finally:
            if close_stream:
                stream.close()

    def object_exists(self, object_name, bucket_name=None):
        bucket_name = bucket_name or self.bucket

        try:
            self.minio_client.stat_object(bucket_name, object_name)
            return True
        except S3Error as error:
            if error.code in {
                "NoSuchKey",
                "NoSuchObject",
                "NoSuchBucket"
            }:
                return False
            raise

    def get_object(self, object_name, bucket_name=None):
        bucket_name = bucket_name or self.bucket
        response = self.minio_client.get_object(bucket_name, object_name)

        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def get_presigned_url(
            self,
            object_name,
            bucket_name=None,
            expires=None
    ):
        bucket_name = bucket_name or self.bucket
        kwargs = {}

        if expires is not None:
            kwargs["expires"] = expires

        return self.minio_client.presigned_get_object(
            bucket_name,
            object_name,
            **kwargs
        )
