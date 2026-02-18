import os, time, random, logging
from urllib.parse import urlparse, parse_qs, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3, requests
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

if __name__== '__main__':
    print('Testing...')
