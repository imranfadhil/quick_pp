"""Exception definition"""

from psycopg2.errors import UniqueViolation
from sqlalchemy.exc import IntegrityError


class BadRequestException(Exception):
    """
    This exception could be raise when receiving unexpected input from user
    """

    def __init__(self, name: str) -> None:
        self.name = name


class DataNotFoundException(Exception):
    """
    This exception could be raise when receiving unexpected input from user
    """

    def __init__(self, name: str) -> None:
        self.name = name


class MethodNotAllowedException(Exception):
    """
    This exception could be raise when receiving unexpected input from user
    """

    def __init__(self, name: str) -> None:
        self.name = name


class NotAuthorizedException(Exception):
    """
    This exception could be raise when receiving unexpected input from user
    """

    def __init__(self, name: str) -> None:
        self.name = name


class UnknownException(Exception):
    """
    This exception could be raise when receiving unexpected input from user
    """

    def __init__(self, name: str) -> None:
        self.name = name


def return_exception_message(exc: Exception) -> str:
    if isinstance(exc, DataNotFoundException):
        return "Not Found!"
    if isinstance(exc, NotAuthorizedException):
        return "Not Authorized!"
    if isinstance(exc, BadRequestException):
        return "Bad request!"
    if isinstance(exc, UnknownException):
        return "Unrecognized error!"
    return "Oops! Something was wrong"


def return_database_integrity_error_message(exc: IntegrityError) -> str:
    if isinstance(exc.orig, UniqueViolation):
        return "Unique Violation! Data input already exists"
    return "Oops! Data Inegrity Error, please check the input data"
