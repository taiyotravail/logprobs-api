"""Quota global d'analyses par jour, pour garder l'hébergement dans l'offre gratuite.

Le compteur vit en mémoire : il est remis à zéro à chaque redémarrage du
serveur. C'est volontaire — un redémarrage n'arrive qu'après une période sans
trafic, et un abus continu garde au contraire le serveur (et le compteur) en vie.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import date, datetime, timezone


def utc_today() -> date:
    """Date du jour en UTC (la remise à zéro du quota se fait à minuit UTC)."""
    return datetime.now(timezone.utc).date()


class DailyQuota:
    """Compteur de requêtes par jour, thread-safe.

    Args:
        limit: nombre max de requêtes par jour. `0` ou moins = illimité.
        today: horloge injectable (pour les tests).
    """

    def __init__(self, limit: int, today: Callable[[], date] = utc_today) -> None:
        self._limit = limit
        self._today = today
        self._lock = threading.Lock()
        self._day = today()
        self._count = 0

    def try_consume(self) -> bool:
        """Réserve une requête sur le quota du jour.

        Returns:
            `True` si la requête est autorisée, `False` si le quota est épuisé.
        """
        if self._limit <= 0:
            return True
        with self._lock:
            current_day = self._today()
            if current_day != self._day:
                self._day, self._count = current_day, 0
            if self._count >= self._limit:
                return False
            self._count += 1
            return True
