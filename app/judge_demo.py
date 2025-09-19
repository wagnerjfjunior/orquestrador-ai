# =============================================================================
# File: app/judge_demo.py
# Purpose: Demo rápido em linha de comando para validar app/judge.py
# Run:
#   python -m app.judge_demo
# =============================================================================

from app.judge import contribution_ratio, judge_answers

PROMPT = "Explique o conceito de Entropia em Termodinâmica de forma simples."

OPENAI_ANS = """A entropia é um conceito fundamental na termodinâmica que mede a desordem
ou a aleatoriedade de um sistema. Em termos simples, podemos pensar na entropia
como uma forma de quantificar o quanto a energia em um sistema está dispersa ou
distribuída. A segunda lei afirma que a entropia total de um sistema isolado
tende a aumentar ao longo do tempo, indicando a direção natural dos processos.
"""

GEMINI_ANS = """Imagine uma sala arrumada (baixa entropia) e, depois de brincar,
uma sala bagunçada (alta entropia). Em termodinâmica, a entropia mede o grau
de desordem. Em sistemas isolados, a entropia tende a aumentar com o tempo,
conforme a Segunda Lei da Termodinâmica. Para reduzir a bagunça, é preciso
gastar energia em outro lugar.
"""

FINAL_FUSED = """Entropia é uma medida de quanta “bagunça” existe no sistema e de quão
espalhada está a energia. A Segunda Lei diz que, em sistemas isolados, a entropia
tende a aumentar, indicando a direção natural dos processos. Exemplos cotidianos
incluem um quarto que bagunça com o tempo e o calor que flui do quente para o frio.
"""

def main():
    print("=== DEMO JUDGE / CONTRIBUTION ===")
    print("Prompt:", PROMPT, "\n")

    sources = {"openai": OPENAI_ANS, "gemini": GEMINI_ANS}
    ratios = contribution_ratio(FINAL_FUSED, sources)

    print("Final fused answer:\n", FINAL_FUSED, "\n")
    print("Contributions (should sum ~ 1.0):")
    for name, frac in ratios.items():
        print(f"  - {name:6s}: {frac:.3f}")
    print()

    verdict = judge_answers(OPENAI_ANS, GEMINI_ANS)
    print("Heuristic judge (A=openai, B=gemini):")
    print(" ", verdict)

if __name__ == "__main__":
    main()
