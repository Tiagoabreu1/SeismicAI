| Estrato | n | Taxa de positivos | ROC-AUC Climatologia | PR-AUC Climatologia | ROC-AUC Combinado | PR-AUC Combinado | ROC-AUC Reg. logística | PR-AUC Reg. logística | ROC-AUC RBM+logreg | PR-AUC RBM+logreg | ROC-AUC MLP (seq.) | PR-AUC MLP (seq.) | ROC-AUC MLP | PR-AUC MLP | ROC-AUC LSTM | PR-AUC LSTM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Todo o teste | 184,340 | 0.166 | 0.830 | 0.528 | 0.836 | 0.552 | 0.839 | 0.565 | 0.836 | 0.554 | 0.839 | 0.566 | 0.841 | 0.569 | 0.841 | 0.571 |
| Sem M ≥ 4 na célula nas últimas 4 semanas | 112,522 | 0.072 | 0.775 | 0.216 | 0.778 | 0.229 | 0.773 | 0.229 | 0.770 | 0.223 | 0.774 | 0.225 | 0.780 | 0.233 | 0.780 | 0.233 |
| Com M ≥ 4 na célula nas últimas 4 semanas | 71,818 | 0.314 | 0.766 | 0.610 | 0.778 | 0.635 | 0.787 | 0.651 | 0.782 | 0.638 | 0.787 | 0.652 | 0.789 | 0.655 | 0.790 | 0.657 |
| Com M ≥ 4 na célula na última semana | 30,682 | 0.410 | 0.763 | 0.691 | 0.777 | 0.715 | 0.795 | 0.736 | 0.788 | 0.720 | 0.795 | 0.739 | 0.797 | 0.741 | 0.798 | 0.743 |
| Com M ≥ 5 na célula nas últimas 4 semanas | 14,812 | 0.425 | 0.748 | 0.682 | 0.764 | 0.703 | 0.796 | 0.745 | 0.786 | 0.720 | 0.795 | 0.746 | 0.798 | 0.751 | 0.799 | 0.753 |
| Com M ≥ 5 nos vizinhos nas últimas 4 semanas | 70,249 | 0.244 | 0.810 | 0.593 | 0.818 | 0.616 | 0.823 | 0.632 | 0.820 | 0.620 | 0.823 | 0.632 | 0.825 | 0.636 | 0.826 | 0.637 |
| Células quietas (taxa histórica abaixo de 3.0%/semana) | 45,760 | 0.017 | 0.629 | 0.025 | 0.662 | 0.039 | 0.665 | 0.068 | 0.663 | 0.063 | 0.671 | 0.065 | 0.679 | 0.070 | 0.679 | 0.068 |
| Células ativas (taxa histórica acima de 20.4%/semana) | 46,020 | 0.433 | 0.685 | 0.640 | 0.712 | 0.672 | 0.719 | 0.682 | 0.711 | 0.669 | 0.718 | 0.683 | 0.720 | 0.685 | 0.721 | 0.687 |
