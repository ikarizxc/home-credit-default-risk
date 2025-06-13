import logging
import sys

def configure_root_logger(
    level: int = logging.INFO,
    fmt: str = '%(asctime)s %(name)s %(levelname)s %(message)s'
) -> None:
    """
    Настраивает корневой логгер для всего приложения.

    Args:
        level (int): Минимальный уровень логирования для корневого логгера.
        fmt (str): Формат вывода сообщений.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Удаляем все старые хендлеры, чтобы конфигурировать чисто
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    root.addHandler(handler)
