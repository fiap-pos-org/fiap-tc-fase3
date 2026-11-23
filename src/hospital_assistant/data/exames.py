"""Catálogo de exames laboratoriais com faixas de referência e valores críticos (PROT-VCRIT-01 §2)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExameLab:
    nome: str
    unidade: str
    referencia: tuple[float, float]
    critico_abaixo: float | None = None  # crítico se valor < limite
    critico_acima: float | None = None  # crítico se valor > limite
    critico_acima_inclusivo: bool = False  # crítico se valor >= limite

    def is_critico(self, valor: float) -> bool:
        if self.critico_abaixo is not None and valor < self.critico_abaixo:
            return True
        if self.critico_acima is None:
            return False
        if self.critico_acima_inclusivo:
            return valor >= self.critico_acima
        return valor > self.critico_acima


EXAMES_LAB: dict[str, ExameLab] = {
    "potassio": ExameLab("Potássio", "mEq/L", (3.5, 5.0), 2.8, 6.2),
    "sodio": ExameLab("Sódio", "mEq/L", (135, 145), 120, 160),
    "glicose": ExameLab("Glicose", "mg/dL", (70, 99), 50, 450),
    "hemoglobina": ExameLab("Hemoglobina", "g/dL", (12.0, 16.0), 7.0),
    "plaquetas": ExameLab("Plaquetas", "/mm³", (150_000, 450_000), 20_000),
    "leucocitos": ExameLab("Leucócitos", "/mm³", (4_000, 11_000), 1_000, 50_000),
    "lactato": ExameLab("Lactato", "mmol/L", (0.5, 2.0), None, 4.0, critico_acima_inclusivo=True),
    "inr": ExameLab("INR", "", (0.8, 1.2), None, 5.0),
    "ph": ExameLab("pH arterial", "", (7.35, 7.45), 7.20, 7.60),
    "troponina": ExameLab("Troponina ultrassensível", "ng/L", (0, 14), None, 14),
    "creatinina": ExameLab("Creatinina", "mg/dL", (0.6, 1.2)),
    "ureia": ExameLab("Ureia", "mg/dL", (15, 45)),
    "bicarbonato": ExameLab("Bicarbonato", "mEq/L", (22, 28)),
    "pcr": ExameLab("Proteína C reativa", "mg/L", (0, 5)),
}

# Exames não laboratoriais (imagem, culturas, traçados): só têm status e laudo.
EXAMES_OUTROS = (
    "Hemocultura (2 pares)",
    "Urocultura",
    "Radiografia de tórax",
    "Eletrocardiograma",
    "Tomografia de crânio sem contraste",
    "Ultrassonografia de rins e vias urinárias",
)
