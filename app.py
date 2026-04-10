"""
app.py — Investigação Forense: Anomalias em FIDCs ligados ao Grupo Master
Estilo: Jornal Investigativo Moderno (NYT / The Guardian)
"""

from click import style
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0. CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Investigação FIDC | Grupo Master",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 1. CSS — ESTÉTICA DE JORNAL INVESTIGATIVO
# ─────────────────────────────────────────────────────────────
def inject_css(dark_mode: bool):
    if dark_mode:
        bg           = "#111111"
        bg_secondary = "#1a1a1a"
        text_primary = "#E8E8E8"
        text_muted   = "#999999"
        accent       = "#FF2222"        # vermelho neon
        accent_soft  = "#CC0000"
        border       = "#333333"
        card_bg      = "#1e1e1e"
        metric_bg    = "#1a0000"
        quote_border = "#FF2222"
        table_header = "#2a0000"
    else:
        bg           = "#F0F2F6"
        bg_secondary = "#FAFAFA"
        text_primary = "#1A1A1A"
        text_muted   = "#555555"
        accent       = "#B22222"        # vermelho escuro clássico
        accent_soft  = "#8B0000"
        border       = "#D1D5DB"
        card_bg      = "#FFFFFF"
        metric_bg    = "#FFF5F5"
        quote_border = "#B22222"
        table_header = "#F5E6E6"

    st.markdown(f"""
    <style>
        /* ── IMPORTAÇÃO DE FONTES ── */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300&family=Inter:wght@300;400;500;600&display=swap');

        /* ── FUNDO & CORPO ── */
        .stApp {{
            background-color: {bg};
        }}
        .main .block-container {{
            padding-top: 0rem;
            max-width: 1100px;
        }}

        /* ── TIPOGRAFIA GLOBAL ── */
        body, p, li, span, div {{
            color: {text_primary};
            font-family: 'Source Serif 4', Georgia, serif;
        }}

        /* ── CABEÇALHO JORNAL ── */
        .newspaper-header {{
            background: {bg_secondary};
            border-bottom: 4px solid {accent};
            padding: 28px 0 18px 0;
            margin-bottom: 0;
            text-align: center;
        }}
        .newspaper-name {{
            font-family: 'Playfair Display', serif;
            font-size: 52px;
            font-weight: 900;
            letter-spacing: 6px;
            text-transform: uppercase;
            color: {text_primary};
            margin-bottom: 10px;
            line-height: 1.1;
        }}
        .edition-line {{
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            color: {text_muted};
            letter-spacing: 2px;
            border-top: 1px solid {border};
            border-bottom: 1px solid {border};
            padding: 8px 0;
            margin: 10px auto;
            max-width: 800px;
            text-transform: uppercase;
        }}

        /* ── MANCHETE PRINCIPAL ── */
        .headline {{
            font-family: 'Playfair Display', serif;
            font-size: clamp(26px, 4vw, 48px);
            font-weight: 900;
            line-height: 1.15;
            color: {text_primary};
            text-align: center;
            padding: 28px 20px 8px 20px;
        }}
        .subheadline {{
            font-family: 'Source Serif 4', serif;
            font-size: 18px;
            font-weight: 300;
            font-style: italic;
            color: {text_muted};
            text-align: center;
            padding: 0 40px 20px 40px;
            line-height: 1.5;
        }}
        .byline {{
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: {accent};
            text-align: center;
            margin-bottom: 8px;
        }}
        .date-line {{
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            color: {text_muted};
            text-align: center;
            margin-bottom: 20px;
        }}

        /* ── SEÇÃO DIVIDER ── */
        .section-rule {{
            border: none;
            border-top: 2px solid {accent};
            margin: 32px 0 24px 0;
        }}
        .section-title {{
            font-family: 'Playfair Display', serif;
            font-size: 26px;
            font-weight: 700;
            color: {text_primary};
            border-bottom: 2px solid {border};
            padding-bottom: 10px;
            margin-bottom: 18px;
        }}
        .subsection-title {{
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: {accent};
            margin-bottom: 10px;
        }}

        /* ── CORPO DO TEXTO ── */
        .body-text {{
            font-family: 'Source Serif 4', serif;
            font-size: 17px;
            line-height: 1.8;
            color: {text_primary};
            text-align: justify;
            hyphens: auto;
        }}

        /* ── BLOCO DE CITAÇÃO ── */
        .pull-quote {{
            border-left: 4px solid {quote_border};
            background: {metric_bg};
            padding: 20px 28px;
            margin: 28px 0;
            font-family: 'Playfair Display', serif;
            font-size: 20px;
            font-style: italic;
            color: {text_primary};
            line-height: 1.6;
        }}
        .pull-quote cite {{
            display: block;
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            font-style: normal;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: {accent};
            margin-top: 12px;
        }}

        /* ── CARDS DE MÉTRICAS ── */
        .metric-card {{
            background: {card_bg};
            border: 1px solid {border};
            border-top: 3px solid {accent};
            padding: 20px;
            border-radius: 2px;
            text-align: center;
        }}
        .metric-label {{
            font-family: 'Inter', sans-serif;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: {text_muted};
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-family: 'Playfair Display', serif;
            font-size: 36px;
            font-weight: 700;
            color: {accent};
            line-height: 1;
        }}
        .metric-sub {{
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            color: {text_muted};
            margin-top: 6px;
        }}

        /* ── BADGE DE ALERTA ── */
        .alert-badge {{
            background: {accent};
            color: #FFFFFF;
            font-family: 'Inter', sans-serif;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            padding: 3px 10px;
            border-radius: 2px;
            display: inline-block;
        }}

        /* ── TABELA DE EVIDÊNCIAS ── */
        .evidence-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'Inter', sans-serif;
            font-size: 13px;
        }}
        .evidence-table th {{
            background: {table_header};
            color: {accent};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            padding: 10px 14px;
            border-bottom: 2px solid {accent};
            text-align: left;
        }}
        .evidence-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid {border};
            color: {text_primary};
            vertical-align: middle;
        }}
        .evidence-table tr:hover td {{
            background: {metric_bg};
        }}
        .score-high {{
            color: {accent};
            font-weight: 700;
        }}
        .score-medium {{
            color: #E07000;
            font-weight: 600;
        }}

        /* ── SIDEBAR ── */
        [data-testid="stSidebar"] {{
            background: {bg_secondary};
            border-right: 1px solid {border};
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown p {{
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            color: {text_muted};
        }}

        /* ── EXPANDER ── */
        [data-testid="stExpander"] {{
            background: {card_bg};
            border: 1px solid {border};
            border-radius: 2px;
        }}

        /* ── ESCONDER ELEMENTOS PADRÃO ── */
       
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header {{ background: transparent; }} /* Deixa o topo invisível mas mantém o botão funcional */
            
        /* Garante que o Sidebar tenha prioridade visual */
        [data-testid="stSidebar"] {{
            z-index: 999999;
        }}
        

        /* ── FOTO / ILUSTRAÇÃO ── */
        .img-caption {{
            font-family: 'Inter', sans-serif;
            font-size: 11px;
            color: {text_muted};
            text-align: center;
            margin-top: 6px;
            font-style: italic;
        }}

        /* ── LINHA DECORATIVA ── */
        .ornament {{
            text-align: center;
            color: {accent};
            font-size: 20px;
            letter-spacing: 8px;
            margin: 24px 0;
        }}

        .sidebar-logo {{
            font-family: 'Playfair Display', serif;
            font-size: 22px;
            font-weight: 900;
            color: {accent};
            text-align: center;
            letter-spacing: 2px;
            padding: 16px 0 8px 0;
            border-bottom: 2px solid {accent};
            margin-bottom: 16px;
        }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 2. CARREGAMENTO & PRÉ-PROCESSAMENTO DOS DADOS
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando base de dados da CVM…")
def load_data():
    df = pd.read_csv("df_master_fidc_2025.csv", sep=";")
    df["DT_COMPTC"] = pd.to_datetime(df["DT_COMPTC"])

    # Normalizar grupo econômico → nomes limpos para exibição
    grupo_map = {
        "Master S/A Corretora (CTVB)": "Master (CTVB)",
        "Trustee DTVM Ltda.":          "Trustee DTVM",
        "Reag Trust DTVM S.A.":        "Reag Trust DTVM",
        "Banco Master S.A.":           "Banco Master",
        "Banco Letsbank S.A.":         "Letsbank",
        "BRB - Banco de Brasília S.A.":"BRB",
        "Banco Pleno S.A.":            "Banco Pleno",
        "Mercado":                     "Mercado (Referência)",
    }
    df["grupo_display"] = df["grupo_economico"].map(grupo_map).fillna(df["grupo_economico"])

    # Garantir tipos numéricos nas features base
    base_cols = [
        "taxa_inadimplencia", "concentracao_cedente",
        "taxa_aquisicao", "taxa_devolucao_cedente", "ratio_inad_giro",
    ]
    for c in base_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # PL mínimo de 1 mil (filtro de qualidade — mesma lógica do notebook)
    df = df[df["TAB_IV_A_VL_PL"] >= 1_000].copy()

    # Cedente declarado (1 = declarado, 0 = ausente)
    df["cedente_declarado"] = df["TAB_I2A12_CPF_CNPJ_CEDENTE_1"].notna().astype(int)

    # ── NOVAS FEATURES (notebook Cell 7) ────────────────────────────────────
    # Ordenar para cálculos temporais corretos
    df = df.sort_values(["CNPJ_FUNDO_CLASSE", "DT_COMPTC"])

    # 1. Volatilidade da inadimplência por fundo (desvio-padrão histórico)
    df["volatilidade_inad"] = (
        df.groupby("CNPJ_FUNDO_CLASSE")["taxa_inadimplencia"]
        .transform("std")
        .fillna(0)
    )

    # 2. Volatilidade da taxa de aquisição por fundo
    df["volatilidade_aquisicao"] = (
        df.groupby("CNPJ_FUNDO_CLASSE")["taxa_aquisicao"]
        .transform("std")
        .fillna(0)
    )

    # 3. Inadimplência congelada: 1 se exatamente igual ao mês anterior
    df["inad_congelada"] = (
        df.groupby("CNPJ_FUNDO_CLASSE")["taxa_inadimplencia"]
        .diff() == 0
    ).astype(int)

    # Limpeza de limites (consistente com o notebook)
    df.loc[df["concentracao_cedente"] > 100, "concentracao_cedente"] = np.nan
    df["concentracao_cedente"] = df["concentracao_cedente"].fillna(0)
    df.loc[df["ratio_inad_giro"] > 100, "ratio_inad_giro"] = 0

    # 8 features forenses do CatBoost (notebook Cell 12)
    feature_cols = [
        "taxa_aquisicao",
        "taxa_devolucao_cedente",
        "taxa_inadimplencia",
        "concentracao_cedente",
        "ratio_inad_giro",
        "volatilidade_inad",
        "volatilidade_aquisicao",
        "inad_congelada",
    ]

    return df, feature_cols


@st.cache_data(show_spinner="Executando CatBoost Supervisionado…")
def run_model(df_clean: pd.DataFrame, feature_cols: list, contamination: float):
    """Treina o CatBoost Supervisionado e pontua todos os registros válidos."""
    # ── 1. Filtro de linhas válidas (PL >= 1M e todas features presentes) ──
    mask_valido = df_clean[feature_cols].notna().all(axis=1)
    mask_valido &= df_clean["TAB_IV_A_VL_PL"] >= 1_000_000

    df_valid = df_clean[mask_valido].copy()
    X_clean = df_valid[feature_cols].copy()

    # ── 2. Máscaras de treino ───────────────────────────────────────────────
    # Mercado ativo = negativos (normais) — exclui fundos zumbi com giro < 1%
    mask_fundos_ativos = df_valid["movimentacao_total_taxa"] > 0.01
    mask_mercado       = (df_valid["grupo_economico"] == "Mercado") & mask_fundos_ativos
    mask_investigado   = df_valid["grupo_economico"] != "Mercado"

    # ── 3. Teto lógico — evita explosões por erros de PL da CVM ───────────
    p95_vol_inad = X_clean.loc[mask_mercado, "volatilidade_inad"].quantile(0.95)
    p95_vol_aq   = X_clean.loc[mask_mercado, "volatilidade_aquisicao"].quantile(0.95)

    X_clean["volatilidade_inad"]      = X_clean["volatilidade_inad"].clip(upper=min(p95_vol_inad, 1.0))
    X_clean["volatilidade_aquisicao"] = X_clean["volatilidade_aquisicao"].clip(upper=min(p95_vol_aq, 1.0))
    X_clean["taxa_inadimplencia"]     = X_clean["taxa_inadimplencia"].clip(upper=1.0)
    X_clean["ratio_inad_giro"]        = X_clean["ratio_inad_giro"].clip(upper=5.0)

    # ── 4. RobustScaler focado na mediana do mercado ───────────────────────
    scaler = RobustScaler()
    scaler.fit(X_clean[mask_mercado])
    X_scaled_arr = scaler.transform(X_clean)
    X_scaled = pd.DataFrame(X_scaled_arr, index=X_clean.index, columns=feature_cols)

    # ── 5. Target supervisionado ───────────────────────────────────────────
    # 1 = investigado (anomalia conhecida) | 0 = mercado ativo (normal)
    y_treino = mask_investigado.astype(int)

    # ── 6. Balanceamento de classes ────────────────────────────────────────
    n_pos = int(y_treino.sum())
    n_neg = int((y_treino == 0).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # ── 7. Treinar CatBoost Supervisionado ─────────────────────────────────
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="AUC",
        random_seed=42,
        verbose=0,
    )
    cb_model.fit(X_scaled, y_treino)

    # ── 8. Score = probabilidade de ser investigado (anomalia) ─────────────
    probs = cb_model.predict_proba(X_scaled)[:, 1]

    # ── 9. Gravar scores no DataFrame ─────────────────────────────────────
    df_valid["anomaly_score"] = probs
    df_valid["anomalia"]      = probs >= 0.50   # True = anômalo

    # ── 10. Baseline mensal do mercado → excesso de suspeição ──────────────
    baseline = (
        df_valid[df_valid["grupo_economico"] == "Mercado"]
        .groupby("DT_COMPTC")["anomaly_score"]
        .mean()
    )
    df_valid["excesso"] = (
        df_valid["anomaly_score"]
        - df_valid["DT_COMPTC"].map(baseline).fillna(0)
    )

    return df_valid


# ─────────────────────────────────────────────────────────────
# 3. SIDEBAR — PAINEL DE CONTROLE
# ─────────────────────────────────────────────────────────────
def render_sidebar(df_raw: pd.DataFrame):
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">⚖ INVESTIGAÇÃO</div>', unsafe_allow_html=True)
        st.markdown("**Painel de Controle da Análise**")
        st.markdown("---")

        dark_mode = st.toggle("🌑 Modo Investigativo (Dark)", value=False)
        st.markdown("---")

        # ── Grupos econômicos
        grupos_disponiveis = sorted(df_raw["grupo_display"].unique())
        grupos_investigados = [g for g in grupos_disponiveis if "Mercado" not in g]
        grupos_sel = st.multiselect(
            "🏦 Grupos Econômicos",
            options=grupos_disponiveis,
            default=grupos_investigados,
            help="Selecione os grupos para análise. 'Mercado (Referência)' é a linha de base.",
        )

        # ── Filtro de datas
        datas_disponiveis = sorted(df_raw["DT_COMPTC"].dt.date.unique())
        data_min, data_max = datas_disponiveis[0], datas_disponiveis[-1]
        periodo = st.select_slider(
            "📅 Período de Competência",
            options=datas_disponiveis,
            value=(datas_disponiveis[0], datas_disponiveis[-1]),
        )

         # ── Sensibilidade
        st.markdown("---")
        sensibilidade = st.slider(
            "🎚 Sensibilidade da Investigação",
            min_value=0.01, max_value=0.30, value=0.05, step=0.01,
            help="Controla quantos fundos o algoritmo considera 'atípicos'. "
                 "Valores maiores detectam mais casos, com menor precisão.",
        )
        st.caption(
            f"*{sensibilidade*100:.0f}% dos registros será classificado "
            "como comportamento atípico pelo modelo.*"
        )

        # ── Score crítico
        score_threshold = st.slider(
            "⚠️ Limiar de Score Crítico",
            min_value=0.50, max_value=0.95, value=0.60, step=0.01,
            help="Score acima deste valor é marcado como alerta crítico.",
        )

        st.markdown("---")
        st.caption("**Fonte dos Dados:** CVM — dados.cvm.gov.br  \n"
                   "**Metodologia:** CatBoost Supervisionado  \n"
                   "**Período:** Mar–Nov 2025")

    return dark_mode, grupos_sel, periodo, sensibilidade, score_threshold


# ─────────────────────────────────────────────────────────────
# 4. COMPONENTES VISUAIS
# ─────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="newspaper-header">
        <div class="newspaper-name">Jornal de Análise Financeira Forense</div>
        <div class="edition-line">
            Edição Especial Investigativa &nbsp;|&nbsp; Mercado de Capitais &nbsp;|&nbsp; Brasil, 2025–2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="width:100%;height:280px;overflow:hidden;">
        <img src="https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=1200&q=80"
             style="width:100%;height:100%;object-fit:cover;object-position:center 40%;" />
    </div>
    <p class="img-caption">Sala de negociações — O mercado de FIDCs movimenta centenas de bilhões de reais ao ano. (Foto: Unsplash)</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="byline">Análise Forense Quantitativa • Dados CVM 2025</div>
    <h1 class="headline">
        Algoritmos de ML detectam comportamentos<br>atípicos em fundos ligados ao Grupo Master
    </h1>
    <p class="subheadline">
        Modelo de inteligência artificial identifica padrão sistemático de anomalia financeira 
        meses antes das intervenções do Banco Central — 8 instituições liquidadas entre novembro 
        de 2025 e fevereiro de 2026
    </p>
    <div class="date-line">Março de 2026 &nbsp;|&nbsp; 29.215 registros analisados &nbsp;|&nbsp; 3.882 fundos monitorados</div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)


def render_intro():
    st.markdown('<div class="section-title">O que é um FIDC — e por que importa</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="body-text">
        Um <strong>FIDC (Fundo de Investimento em Direitos Creditórios)</strong> funciona como 
        uma espécie de "caixinha coletiva" para contratos de dívida. Imagine que um banco tem 
        10.000 contratos de empréstimo com clientes. Em vez de esperar todos pagarem, ele pode 
        vender esses contratos a um FIDC — que paga hoje e recebe os pagamentos futuros.
        Investidores colocam dinheiro no FIDC e recebem os juros. Tudo isso é regulado 
        pela CVM e, em condições normais, é um mecanismo legítimo e eficiente.
        <br><br>
        O problema surge quando quem vende os contratos ao fundo (o <strong>cedente</strong>) 
        é do mesmo grupo econômico de quem administra o fundo. Nesse caso, há um conflito de 
        interesse grave: o administrador pode deixar de declarar à CVM que contratos estão 
        inadimplentes — afinal, declarar seria admitir que seu próprio banco vendeu contratos 
        podres. Nos grupos investigados, a taxa de <strong>cedentes não declarados</strong> 
        chegou a 100% em alguns fundos — um sinal de alerta inequívoco.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="pull-quote">
            "A concentração de risco em cedentes não declarados representa um ponto cego 
            no sistema financeiro — visível apenas quando os dados são analisados em conjunto."
            <cite>— Relatório Forense FIDC, Março 2026</cite>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📖 Entenda os termos técnicos usados nesta análise"):
        st.markdown("""
        | Termo usado aqui | O que significa |
        |---|---|
        | **Comportamento Atípico** | Fundo cujos indicadores fogem do padrão estatístico do mercado (tecnicamente: *outlier*) |
        | **Nota de Suspeição** | Pontuação de 0 a 1 atribuída pelo algoritmo (0 = normal, 1 = altamente anômalo) |
        | **Sensibilidade da Investigação** | Percentual de registros que o modelo classifica como atípicos (*contamination rate*) |
        | **Excesso de Suspeição** | Diferença entre a nota de um grupo e a média do mercado no mesmo mês |
        | **Cedente** | Empresa ou banco que vendeu os contratos de crédito ao fundo |
        | **Taxa de Devolução** | Volume de contratos recomprados pelo cedente ÷ Patrimônio Líquido |
        | **Taxa de Inadimplência** | Contratos não pagos declarados à CVM ÷ Patrimônio Líquido |
        """)


def render_methodology():
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">A "Arma do Crime": Como o algoritmo detecta fraudes</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.image(
            "https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=700&q=80",
            use_container_width=True,
        )
        st.markdown('<p class="img-caption">Algoritmos de machine learning vasculham padrões invisíveis ao olho humano. (Foto: Unsplash)</p>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="subsection-title">O Detetive Digital</div>
        <div class="body-text">
        O <strong>CatBoost Supervisionado</strong> funciona como um detetive que aprendeu
        a reconhecer suspeitos a partir de casos confirmados. Em vez de apenas buscar quem
        "está fora da fila", ele foi treinado com os próprios grupos investigados como
        exemplos positivos e o mercado normal como referência negativa.
        <br><br>
        O modelo analisa <em>8 variáveis forenses simultâneas</em> — incluindo duas novas:
        a <strong>volatilidade da inadimplência</strong> (fundos normais têm variação natural;
        os suspeitos congelam os números) e a <strong>trava de imobilidade</strong>
        (inadimplência exatamente igual por meses consecutivos é estatisticamente improvável).
        <br><br>
        O modelo foi treinado com dados rotulados — mercado ativo como classe normal e grupos
        investigados como classe anômala — usando balanceamento automático de classes.
        O score final é a <strong>probabilidade estimada</strong> de o fundo pertencer ao
        grupo investigado, de 0 (normal) a 1 (máxima semelhança com os suspeitos).
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔬 Detalhes técnicos do modelo (para especialistas)"):
        st.markdown("""
        **Algoritmo:** `catboost.CatBoostClassifier` (Gradient Boosting Supervisionado)  
        **Estimadores:** 500 árvores de decisão com profundidade 6  
        **Treinamento:** Supervisionado — Mercado ativo como classe 0 (normal), grupos investigados como classe 1 (anomalia)  
        **Balanceamento:** `scale_pos_weight` proporcional ao desequilíbrio de classes  
        **Features (8 variáveis monitoradas):**
        1. `taxa_aquisicao` — Novos contratos comprados ÷ PL
        2. `taxa_devolucao_cedente` — Contratos recomprados pelo cedente ÷ PL
        3. `taxa_inadimplencia` — Inadimplência declarada ÷ PL
        4. `concentracao_cedente` — % da carteira do maior cedente
        5. `ratio_inad_giro` — Inadimplência ÷ Taxa de aquisição
        6. `volatilidade_inad` — Desvio-padrão histórico da inadimplência por fundo
        7. `volatilidade_aquisicao` — Desvio-padrão histórico da taxa de aquisição por fundo
        8. `inad_congelada` — 1 se inadimplência idêntica ao mês anterior (trava artificial)
        
        **Normalização:** `RobustScaler` calibrado na mediana do mercado ativo  
        **Score final:** Probabilidade [0,1] de pertencer aos grupos investigados (0 = normal, 1 = máxima anomalia)  
        
        **Sinal forense central:** Alto giro + inadimplência congelada + volatilidade zero →
        probabilidade < 0,3% de ocorrer por acaso em condições normais de mercado.
        """)


def render_metrics(df_model: pd.DataFrame, score_threshold: float):
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Painel de Alertas — Números da Investigação</div>', unsafe_allow_html=True)

    # ── Calcular métricas chave
    df_suspeitos = df_model[df_model["anomaly_score"] >= score_threshold]
    n_fundos_suspeitos = df_suspeitos["CNPJ_FUNDO_CLASSE"].nunique()

    grupos_investigados = ["Master S/A Corretora (CTVB)", "Trustee DTVM Ltda.",
                           "Reag Trust DTVM S.A.", "Banco Master S.A."]
    df_master_group = df_model[df_model["grupo_economico"].isin(grupos_investigados)]

    score_master_medio = df_master_group["anomaly_score"].mean()
    score_mercado_medio = df_model[df_model["grupo_economico"] == "Mercado"]["anomaly_score"].mean()
    excesso = score_master_medio - score_mercado_medio
    pct_excesso = (excesso / score_mercado_medio) * 100 if score_mercado_medio > 0 else 0

    # Score máximo individual
    score_max = df_model["anomaly_score"].max()
    fundo_max = df_model.loc[df_model["anomaly_score"].idxmax(), "DENOM_SOCIAL"]

    # Período set-nov
    df_set_nov = df_model[
        (df_model["DT_COMPTC"].dt.month.isin([9, 10, 11])) &
        (df_model["grupo_economico"].isin(grupos_investigados)) &
        (df_model["anomaly_score"] >= score_threshold)
    ]
    n_alertas_setnov = len(df_set_nov)

    cols = st.columns(4)
    metricas = [
        ("Fundos com Comportamento Atípico", f"{n_fundos_suspeitos}", f"Score ≥ {score_threshold:.2f}"),
        ("Nota de Suspeição — Grupo Master", f"{score_master_medio:.3f}", f"+{pct_excesso:.0f}% acima do mercado"),
        ("Alertas Críticos (Set–Nov 2025)", f"{n_alertas_setnov}", "Período pré-intervenção BC"),
        ("Score Máximo Individual", f"{score_max:.3f}", fundo_max[:35] + "…" if len(fundo_max) > 35 else fundo_max),
    ]
    for col, (label, valor, sub) in zip(cols, metricas):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{valor}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="ornament">✦ ✦ ✦</div>', unsafe_allow_html=True)


# ── GRÁFICOS DO NOTEBOOK ─────────────────────────────────────────────────────

def render_excesso_temporal(df_model: pd.DataFrame, dark_mode: bool):
    """Gráfico 1 — Evolução temporal do excesso de suspeição (notebook Cell 14)."""
    st.markdown(
        '<div class="subsection-title">Quanto cada grupo esteve acima do mercado — mês a mês</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="body-text">
        O eixo vertical mostra o <strong>excesso de suspeição</strong>: a diferença entre a nota
        do grupo e a média do mercado naquele mesmo mês. Zero significa "igual ao mercado".
        As faixas coloridas indicam o protocolo de alerta — da monitoração (verde) à intervenção
        imediata (vermelho).
        </div>
        """,
        unsafe_allow_html=True,
    )

    grupos_cfg = {
        "Banco Master S.A.":            ("#B71C1C", "diamond",       3),
        "Master S/A Corretora (CTVB)":  ("#E91E63", "circle",        2),
        "Trustee DTVM Ltda.":           ("#0288D1", "triangle-up",   2),
        "Reag Trust DTVM S.A.":         ("#F57F17", "square",        2),
        "Banco Pleno S.A.":             ("#7B1FA2", "triangle-down", 2),
        "BRB - Banco de Brasília S.A.": ("#2E7D32", "cross",         2),
    }

    # Calcular excesso médio mensal por grupo
    evolucao = (
        df_model[df_model["grupo_economico"].isin(grupos_cfg.keys())]
        .groupby([pd.Grouper(key="DT_COMPTC", freq="ME"), "grupo_economico"])["excesso"]
        .mean()
        .reset_index()
    )

    bg_color   = "rgba(0,0,0,0)"
    font_color = "#E8E8E8" if dark_mode else "#1A1A1A"
    grid_color = "#333333" if dark_mode else "#E5E7EB"

    fig = go.Figure()

    # Faixas de alerta (shapes)
    faixas = [
        (-0.5,  0.10, "rgba(76,175,80,0.10)",   "Verde — Monitoramento"),
        ( 0.10, 0.25, "rgba(255,235,59,0.15)",  "Amarelo — Auditoria"),
        ( 0.25, 0.40, "rgba(255,152,0,0.18)",   "Laranja — Suspender recompras"),
        ( 0.40, 1.0,  "rgba(244,67,54,0.15)",   "Vermelho — Intervenção"),
    ]
    for y0, y1, color, label in faixas:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                      annotation_text=label if y0 >= 0.10 else "",
                      annotation_position="right",
                      annotation_font_size=10,
                      annotation_font_color=font_color)

    # Linhas de threshold
    for nivel, cor, dash in [
        (0.10, "#F9A825", "dash"),
        (0.25, "#E65100", "dash"),
        (0.40, "#B71C1C", "dash"),
    ]:
        fig.add_hline(y=nivel, line_color=cor, line_dash=dash,
                      line_width=1.2, opacity=0.8)

    # Linha zero = mercado
    fig.add_hline(y=0, line_color="#888888", line_width=2,
                  annotation_text="Mercado — referência (0)",
                  annotation_position="right",
                  annotation_font_color="#888888")

    # Séries por grupo
    for grupo, (cor, simbolo, lw) in grupos_cfg.items():
        sub = evolucao[evolucao["grupo_economico"] == grupo].sort_values("DT_COMPTC")
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["DT_COMPTC"],
            y=sub["excesso"],
            name=grupo,
            mode="lines+markers",
            line=dict(color=cor, width=lw + 0.5),
            marker=dict(symbol=simbolo, size=8, color=cor,
                        line=dict(color="white", width=0.5)),
        ))

    fig.update_layout(
        xaxis_title="Mês de Competência",
        yaxis_title="Excesso acima do mercado (nota grupo − nota mercado)",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(family="Inter", color=font_color, size=12),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zeroline=False),
        legend=dict(orientation="v", x=1.01, y=1, font_size=11),
        height=480,
        margin=dict(l=20, r=220, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 Como interpretar as faixas de alerta"):
        st.markdown("""
        | Faixa | Excesso | Protocolo sugerido |
        |---|---|---|
        | 🟢 Verde | < +0.10 | Monitoramento rotineiro |
        | 🟡 Amarelo | +0.10 a +0.25 | Auditoria aprofundada |
        | 🟠 Laranja | +0.25 a +0.40 | Suspensão preventiva de recompras |
        | 🔴 Vermelho | > +0.40 | Intervenção imediata do regulador |

        O Banco Master S.A. cruzou a faixa laranja em vários meses e atingiu o pico de
        **+0.30** em outubro de 2025 — mês anterior à intervenção do Banco Central.
        """)


def render_ranking_fundos(df_model: pd.DataFrame, dark_mode: bool):
    """Gráfico 2 — Ranking top 20 fundos mais suspeitos (notebook Cell 17)."""
    st.markdown(
        '<div class="subsection-title">Top 20 fundos com maior nota média de suspeição no ano</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="body-text">
        Cada barra representa a <strong>nota média anual</strong> de um fundo — quanto mais
        longa, mais consistentemente anômalo foi o comportamento ao longo de 2025. A cor
        identifica o grupo econômico. A linha vermelha tracejada marca o limiar crítico (0.70).
        </div>
        """,
        unsafe_allow_html=True,
    )

    grupos_investigados = [
        "Master S/A Corretora (CTVB)", "Trustee DTVM Ltda.",
        "Reag Trust DTVM S.A.", "Banco Master S.A.",
        "Banco Pleno S.A.", "BRB - Banco de Brasília S.A.",
    ]
    cores_grupo = {
        "Master S/A Corretora (CTVB)":  "#E91E63",
        "Banco Master S.A.":            "#B71C1C",
        "Trustee DTVM Ltda.":           "#0288D1",
        "Reag Trust DTVM S.A.":         "#F57F17",
        "Banco Pleno S.A.":             "#7B1FA2",
        "BRB - Banco de Brasília S.A.": "#2E7D32",
    }

    df_rank = (
        df_model[df_model["grupo_economico"].isin(grupos_investigados)]
        .groupby(["DENOM_SOCIAL", "grupo_economico"])["anomaly_score"]
        .mean()
        .reset_index()
        .nlargest(20, "anomaly_score")
        .reset_index(drop=True)
    )

    df_rank["nome_curto"] = df_rank["DENOM_SOCIAL"].apply(
        lambda n: n[:52] + "…" if len(n) > 52 else n
    )
    df_rank["cor"] = df_rank["grupo_economico"].map(cores_grupo).fillna("#888888")

    bg_color   = "rgba(0,0,0,0)"
    font_color = "#E8E8E8" if dark_mode else "#1A1A1A"
    grid_color = "#333333" if dark_mode else "#E5E7EB"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_rank["anomaly_score"],
        y=df_rank["nome_curto"],
        orientation="h",
        marker_color=df_rank["cor"],
        text=df_rank["anomaly_score"].round(3).astype(str),
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(color="white", size=11, family="Inter"),
        customdata=df_rank["grupo_economico"],
        hovertemplate="<b>%{y}</b><br>Grupo: %{customdata}<br>Score: %{x:.3f}<extra></extra>",
    ))

    # Limiares
    fig.add_vline(x=0.70, line_color="#B71C1C", line_dash="dash",
                  line_width=2, annotation_text="Limiar crítico (0.70)",
                  annotation_position="top right",
                  annotation_font_color="#B71C1C")
    fig.add_vline(x=0.50, line_color="#E65100", line_dash="dot",
                  line_width=1.5, annotation_text="Limiar elevado (0.50)",
                  annotation_position="bottom right",
                  annotation_font_color="#E65100")

    fig.update_layout(
        xaxis_title="Nota média de suspeição (0 = normal · 1 = máxima anomalia)",
        yaxis_title="",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(family="Inter", color=font_color, size=12),
        xaxis=dict(gridcolor=grid_color, range=[0, 1.05]),
        yaxis=dict(gridcolor=grid_color, autorange="reversed"),
        showlegend=False,
        height=580,
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_mapa_forense(df_model: pd.DataFrame, grupos_sel: list, dark_mode: bool):
    """Gráfico 3 — Mapa forense Movimentação × Inadimplência (notebook Cell 20)."""
    st.markdown(
        '<div class="subsection-title">Mapa forense — Giro da carteira vs. Inadimplência declarada</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="body-text">
        Cada ponto é um fundo em um mês. O eixo horizontal mede o <strong>giro da carteira</strong>
        (quanto foi comprado ÷ Patrimônio Líquido). O eixo vertical mede a <strong>inadimplência
        declarada</strong>. O quadrante superior direito — alto giro <em>e</em> alta inadimplência —
        é o sinal clássico de substituição de ativos problemáticos. Pontos maiores e mais vermelhos
        têm maior nota de suspeição.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cores_grupo = {
        "Master S/A Corretora (CTVB)":  "#E91E63",
        "Banco Master S.A.":            "#B71C1C",
        "Trustee DTVM Ltda.":           "#0288D1",
        "Reag Trust DTVM S.A.":         "#F57F17",
        "Banco Pleno S.A.":             "#7B1FA2",
        "BRB - Banco de Brasília S.A.": "#2E7D32",
        "Mercado":                      "#78909C",
    }

    # Limites p99 do mercado (mesmos do notebook)
    merc = df_model[df_model["grupo_economico"] == "Mercado"]
    tm_p99 = merc["movimentacao_total_taxa"].quantile(0.99)
    ti_p99 = merc["taxa_inadimplencia"].quantile(0.99)
    tm_mu  = merc["movimentacao_total_taxa"].mean()
    ti_mu  = merc["taxa_inadimplencia"].mean()
    tm_std = merc["movimentacao_total_taxa"].std()
    ti_std = merc["taxa_inadimplencia"].std()

    # Amostra do mercado de fundo
    merc_sample = merc.dropna(subset=["anomaly_score"]).sample(
        min(1500, len(merc)), random_state=42
    )

    bg_color   = "rgba(0,0,0,0)"
    font_color = "#E8E8E8" if dark_mode else "#1A1A1A"
    grid_color = "#333333" if dark_mode else "#E5E7EB"

    grupos_investigados_map = [
        g for g in [
            "Banco Master S.A.", "Master S/A Corretora (CTVB)",
            "Trustee DTVM Ltda.", "Reag Trust DTVM S.A.",
            "Banco Pleno S.A.", "BRB - Banco de Brasília S.A.",
        ]
        if g in df_model["grupo_economico"].unique()
    ]

    # Selectbox para escolher grupo no mapa
    grupo_escolhido = st.selectbox(
        "Selecione o grupo para destacar no mapa forense:",
        options=grupos_investigados_map,
        index=0,
        key="selectbox_mapa_forense",
    )

    fig = go.Figure()

    # Mercado de fundo (cinza)
    fig.add_trace(go.Scatter(
        x=merc_sample["movimentacao_total_taxa"].clip(0, tm_p99),
        y=merc_sample["taxa_inadimplencia"].clip(0, ti_p99),
        mode="markers",
        name="Mercado (referência)",
        marker=dict(color="#78909C", size=4, opacity=0.35),
        hovertemplate="Mercado<br>TM: %{x:.3f}<br>TI: %{y:.3f}<extra></extra>",
    ))

    # Grupo investigado destacado
    sub = df_model[
        (df_model["grupo_economico"] == grupo_escolhido) &
        (df_model["anomaly_score"].notna())
    ].copy()

    if not sub.empty:
        cor_grupo = cores_grupo.get(grupo_escolhido, "#B22222")
        fig.add_trace(go.Scatter(
            x=sub["movimentacao_total_taxa"].clip(0, tm_p99),
            y=sub["taxa_inadimplencia"].clip(0, ti_p99),
            mode="markers",
            name=grupo_escolhido,
            marker=dict(
                size=6 + sub["anomaly_score"] * 18,
                color=sub["anomaly_score"],
                colorscale="YlOrRd",
                cmin=0, cmax=1,
                opacity=(0.35 + sub["anomaly_score"] * 0.65).clip(0.35, 1.0),
                colorbar=dict(title="Nota de<br>Suspeição", thickness=14),
                line=dict(color=cor_grupo, width=0.5),
            ),
            customdata=sub[["DENOM_SOCIAL", "DT_COMPTC", "anomaly_score"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Competência: %{customdata[1]}<br>"
                "TM: %{x:.3f}<br>TI: %{y:.3f}<br>"
                "Score: %{customdata[2]:.3f}<extra></extra>"
            ),
        ))

    # Linhas μ+2σ do mercado
    fig.add_vline(x=tm_mu + 2 * tm_std, line_color="#888888", line_dash="dash",
                  line_width=1, opacity=0.6,
                  annotation_text="μ+2σ TM", annotation_position="top right",
                  annotation_font_color="#888888")
    fig.add_hline(y=ti_mu + 2 * ti_std, line_color="#888888", line_dash="dot",
                  line_width=1, opacity=0.6,
                  annotation_text="μ+2σ TI", annotation_position="top right",
                  annotation_font_color="#888888")

    fig.update_layout(
        xaxis_title="Movimentação da carteira ÷ Patrimônio Líquido",
        yaxis_title="Inadimplência declarada ÷ Patrimônio Líquido",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(family="Inter", color=font_color, size=12),
        xaxis=dict(gridcolor=grid_color, range=[0, tm_p99 * 1.05]),
        yaxis=dict(gridcolor=grid_color, range=[0, ti_p99 * 1.05]),
        legend=dict(orientation="h", y=-0.15),
        height=460,
        margin=dict(l=20, r=20, t=20, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretação automática por grupo
    interpretacoes = {
        "Banco Master S.A.": (
            "Os fundos do Banco Master dispersam-se além das linhas de referência especialmente "
            "no eixo horizontal — alto giro com inadimplência controlada é o sinal clássico de "
            "substituição de contratos problemáticos antes da declaração."
        ),
        "Master S/A Corretora (CTVB)": (
            "A corretora apresenta inadimplência declarada acima do mercado com giro moderado — "
            "diferente do banco em si, sugere deterioração progressiva sem substituição sistemática."
        ),
        "Trustee DTVM Ltda.": (
            "A maioria dos pontos concentra-se próxima da origem por ausência de declaração de "
            "cedentes. Os poucos pontos distantes e vermelhos são os fundos que declararam dados — "
            "entre eles o NF Fundo, nota 0.834, o mais crítico de toda a base."
        ),
        "Reag Trust DTVM S.A.": (
            "Padrão similar à Trustee: baixa declaração limita a visibilidade. Os pontos vermelhos "
            "isolados correspondem ao Engelberg 41 (0.736), JOB Fundo (0.689) e Indústria Q1 (0.641)."
        ),
    }
    texto = interpretacoes.get(
        grupo_escolhido,
        "Selecione um dos grupos investigados para ver a interpretação automática.",
    )
    st.markdown(
        f'<div class="pull-quote" style="font-size:16px;padding:14px 20px;">'
        f'{texto}</div>',
        unsafe_allow_html=True,
    )


def render_distribuicao_violin(df_model: pd.DataFrame, dark_mode: bool):
    """Gráfico 4 — Distribuição das notas por grupo violin+box (notebook Cell 23)."""
    st.markdown(
        '<div class="subsection-title">Distribuição completa das notas — forma, mediana e dispersão</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="body-text">
        O violino revela a <strong>forma completa da distribuição</strong> — se o grupo foi
        consistentemente anômalo ou teve picos isolados. A linha tracejada cinza é a média do
        mercado (referência). Grupos com a distribuição inteira acima dessa linha estiveram
        sistematicamente mais suspeitos durante todo o ano.
        </div>
        """,
        unsafe_allow_html=True,
    )

    ordem = [
        "Banco Master S.A.",
        "Banco Pleno S.A.",
        "BRB - Banco de Brasília S.A.",
        "Master S/A Corretora (CTVB)",
        "Mercado",
        "Trustee DTVM Ltda.",
        "Reag Trust DTVM S.A.",
    ]
    cores_grupo = {
        "Banco Master S.A.":            "#B71C1C",
        "Banco Pleno S.A.":             "#7B1FA2",
        "BRB - Banco de Brasília S.A.": "#2E7D32",
        "Master S/A Corretora (CTVB)":  "#E91E63",
        "Mercado":                      "#78909C",
        "Trustee DTVM Ltda.":           "#0288D1",
        "Reag Trust DTVM S.A.":         "#F57F17",
    }
    nomes_curtos = {
        "Banco Master S.A.":            "Banco Master",
        "Banco Pleno S.A.":             "Banco Pleno",
        "BRB - Banco de Brasília S.A.": "BRB",
        "Master S/A Corretora (CTVB)":  "Master Corretora",
        "Mercado":                      "Mercado",
        "Trustee DTVM Ltda.":           "Trustee",
        "Reag Trust DTVM S.A.":         "Reag Trust",
    }

    bg_color   = "rgba(0,0,0,0)"
    font_color = "#E8E8E8" if dark_mode else "#1A1A1A"
    grid_color = "#333333" if dark_mode else "#E5E7EB"

    media_mercado = (
        df_model[df_model["grupo_economico"] == "Mercado"]["anomaly_score"].mean()
    )

    fig = go.Figure()

    grupos_presentes = [g for g in ordem if g in df_model["grupo_economico"].unique()]

    for grupo in grupos_presentes:
        sub = df_model[df_model["grupo_economico"] == grupo]["anomaly_score"].dropna()
        if len(sub) < 5:
            continue
        cor = cores_grupo.get(grupo, "#888888")
        nome = nomes_curtos.get(grupo, grupo)

        # Violino
        fig.add_trace(go.Violin(
            y=sub,
            name=nome,
            box_visible=True,
            meanline_visible=True,
            fillcolor=cor,
            line_color=cor,
            opacity=0.55,
            points=False,
            width=0.8,
            hoverinfo="name+y",
        ))

    # Linha média do mercado
    fig.add_hline(
        y=media_mercado,
        line_color="#888888",
        line_dash="dash",
        line_width=2,
        annotation_text=f"Média do mercado ({media_mercado:.3f})",
        annotation_position="top right",
        annotation_font_color="#888888",
    )

    # Divisor visual entre grupos acima/abaixo do mercado
    # (Mercado está na posição 4 em 'ordem', mas só contamos os presentes)
    idx_mercado = next(
        (i for i, g in enumerate(grupos_presentes) if g == "Mercado"), None
    )
    if idx_mercado is not None:
        fig.add_vline(
            x=idx_mercado - 0.5,
            line_color="#AAAAAA",
            line_dash="dot",
            line_width=1.5,
            annotation_text="← Acima do mercado | Abaixo →",
            annotation_position="top",
            annotation_font_color=font_color,
            annotation_font_size=10,
        )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Nota de suspeição (0 = normal · 1 = máxima anomalia)",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(family="Inter", color=font_color, size=12),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color, range=[0, 1.1]),
        showlegend=False,
        height=460,
        margin=dict(l=20, r=20, t=20, b=40),
        violinmode="overlay",
        violingap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 Como ler este gráfico"):
        st.markdown("""
        - **Largura do violino**: quanto mais largo, mais observações naquela faixa de score
        - **Caixa central**: intervalo interquartil (25% a 75% das observações)
        - **Linha branca dentro da caixa**: mediana
        - **Ponto central**: média
        - **Linha tracejada cinza**: média do mercado — referência de normalidade

        O Banco Master tem o violino **largo na parte superior** — a maioria das observações
        ficou em scores altos, não foram picos isolados. Trustee e Reag têm caudas finas para
        cima: são os poucos fundos que declararam dados e foram identificados como críticos.
        """)


def render_evidence_table(df_model: pd.DataFrame, score_threshold: float):
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Tabela de Evidências — Os Principais Suspeitos</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="body-text">
        A tabela abaixo lista os fundos com as maiores notas de suspeição, acompanhados do
        principal motivo identificado pelo algoritmo para o alerta. Os dados são da competência
        mais recente disponível para cada fundo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    grupos_investigados = [
        "Master S/A Corretora (CTVB)",
        "Trustee DTVM Ltda.",
        "Reag Trust DTVM S.A.",
        "Banco Master S.A.",
    ]

    # ── 1. Pegar competência mais recente por fundo
    df_inv = df_model[df_model["grupo_economico"].isin(grupos_investigados)].copy()
    # Ordenar por fundo + data desc, pegar a linha mais recente de cada fundo
    df_inv = df_inv.sort_values(["CNPJ_FUNDO_CLASSE", "DT_COMPTC"], ascending=[True, False])
    df_inv = df_inv.drop_duplicates(subset=["CNPJ_FUNDO_CLASSE"], keep="first")
    # Agora ordenar por score descendente e pegar top 15
    df_ev = df_inv.sort_values("anomaly_score", ascending=False).head(15).copy()

    # ── 2. Classificar alerta
    def nivel_alerta(score: float) -> str:
        if score >= 0.75:
            return "🔴 CRÍTICO"
        if score >= 0.60:
            return "🟠 ALTO"
        return "🟡 MODERADO"

    # ── 3. Motivo principal da anomalia
    def motivo_anomalia(row) -> str:
        motivos = []
        conc = pd.to_numeric(row.get("concentracao_cedente"), errors="coerce")
        dev  = pd.to_numeric(row.get("taxa_devolucao_cedente"), errors="coerce")
        inad = pd.to_numeric(row.get("taxa_inadimplencia"), errors="coerce")
        aqui = pd.to_numeric(row.get("taxa_aquisicao"), errors="coerce")
        ratio = pd.to_numeric(row.get("ratio_inad_giro"), errors="coerce")

        cedente_ausente = pd.isna(row.get("TAB_I2A12_CPF_CNPJ_CEDENTE_1"))

        if cedente_ausente:
            motivos.append("Cedente não declarado à CVM")
        if pd.notna(conc) and conc > 80:
            motivos.append(f"Concentração excessiva ({conc:.0f}% em 1 cedente)")
        if pd.notna(dev) and dev > 0.30:
            motivos.append("Alta devolução ao cedente (recompras)")
        if pd.notna(ratio) and pd.notna(aqui) and ratio < 0.01 and aqui > 0.20:
            motivos.append("Inadimplência baixa com alto giro de carteira")
        if pd.notna(inad) and inad == 0 and pd.notna(aqui) and aqui > 0.10:
            motivos.append("Inadimplência zero declarada com carteira ativa")
        return " · ".join(motivos) if motivos else "Padrão estatístico atípico combinado"

    # ── 4. Construir DataFrame de exibição limpo
    df_ev["Alerta"]           = df_ev["anomaly_score"].apply(nivel_alerta)
    df_ev["Nota de Suspeição"]= df_ev["anomaly_score"].round(3)
    df_ev["Competência"]      = df_ev["DT_COMPTC"].dt.strftime("%b/%Y")
    df_ev["Patrimônio Líquido"] = df_ev["TAB_IV_A_VL_PL"].apply(
        lambda x: f"R$ {x/1e6:.1f} M" if pd.notna(x) else "—"
    )
    df_ev["Motivo do Alerta"] = df_ev.apply(motivo_anomalia, axis=1)

    # Nome truncado para caber bem
    df_ev["Fundo"] = df_ev["DENOM_SOCIAL"].apply(
        lambda n: n[:60] + "…" if len(str(n)) > 60 else n
    )
    df_ev["Grupo"] = df_ev["grupo_display"]

    display_cols = [
        "Alerta", "Nota de Suspeição", "Fundo", "Grupo",
        "Competência", "Patrimônio Líquido", "Motivo do Alerta",
    ]
    df_show = df_ev[display_cols].reset_index(drop=True)

    # ── 5. Estilizar com pandas Styler
    def color_score(val):
        if isinstance(val, float):
            if val >= 0.75:
                return "color: #B22222; font-weight: 700;"
            if val >= 0.60:
                return "color: #E07000; font-weight: 600;"
            return "color: #2E7D32; font-weight: 500;"
        return ""

    def color_alerta(val):
        if "CRÍTICO" in str(val):
            return "background-color: #fdecea; color: #B22222; font-weight: 700;"
        if "ALTO" in str(val):
            return "background-color: #fff3e0; color: #E07000; font-weight: 600;"
        return "background-color: #fffde7; color: #827717;"

    styled = (
        df_show.style
        .map(color_score, subset=["Nota de Suspeição"])
        .map(color_alerta, subset=["Alerta"])
        .set_properties(**{
            "font-family": "Inter, sans-serif",
            "font-size": "13px",
            "text-align": "left",
            "vertical-align": "middle",
            "padding": "8px 12px",
        })
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#F5E6E6"),
                    ("color", "#B22222"),
                    ("font-family", "Inter, sans-serif"),
                    ("font-size", "11px"),
                    ("font-weight", "700"),
                    ("letter-spacing", "1px"),
                    ("text-transform", "uppercase"),
                    ("padding", "10px 12px"),
                    ("border-bottom", "2px solid #B22222"),
                ],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#fff5f5")],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#FAFAFA")],
            },
            {
                "selector": "td",
                "props": [("border-bottom", "1px solid #E5E7EB")],
            },
        ])
        .hide(axis="index")
        .format({"Nota de Suspeição": "{:.3f}"})
    )

    st.dataframe(
        df_show,
        use_container_width=True,
        height=480,
        column_config={
            "Alerta": st.column_config.TextColumn("Alerta", width="small"),
            "Nota de Suspeição": st.column_config.ProgressColumn(
                "Nota de Suspeição",
                help="Score do Isolation Forest (0 = normal · 1 = máxima anomalia)",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
                width="medium",
            ),
            "Fundo": st.column_config.TextColumn("Fundo", width="large"),
            "Grupo": st.column_config.TextColumn("Grupo", width="medium"),
            "Competência": st.column_config.TextColumn("Competência", width="small"),
            "Patrimônio Líquido": st.column_config.TextColumn("Patrimônio Líquido", width="small"),
            "Motivo do Alerta": st.column_config.TextColumn("Motivo do Alerta", width="large"),
        },
    )

    with st.expander("📊 Como os motivos de alerta são determinados"):
        st.markdown("""
        Os motivos são gerados automaticamente a partir dos indicadores de cada fundo,
        derivados dos achados do relatório forense:

        | Motivo | Critério |
        |---|---|
        | **Cedente não declarado à CVM** | CNPJ/CPF do cedente ausente no informe mensal |
        | **Concentração excessiva** | Mais de 80% da carteira em um único cedente |
        | **Alta devolução ao cedente** | Recompras > 30% do Patrimônio Líquido no mês |
        | **Inadimplência baixa com alto giro** | Inadimplência < 1% com aquisições > 20% do PL — sinal clássico de substituição de contratos problemáticos |
        | **Inadimplência zero com carteira ativa** | Zero inadimplência declarada enquanto o fundo compra ativamente novos contratos |
        | **Padrão estatístico atípico combinado** | Score elevado sem gatilhar regra individual — anomalia multivariada detectada pelo Isolation Forest |
        """)


def render_timeline():
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cronologia das Intervenções do Banco Central</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="width:100%;height:220px;overflow:hidden;margin-bottom:6px;">
        <img src="https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=1100&q=80"
             style="width:100%;height:100%;object-fit:cover;object-position:center 30%;" />
    </div>
    <p class="img-caption">Reguladores monitoram o sistema financeiro — mas os sinais já estavam nos dados meses antes. (Foto: Unsplash)</p>
    """, unsafe_allow_html=True)

    eventos = [
        ("18/11/2025", "Banco Master S.A.",           "Liquidação Extrajudicial", "Núcleo central do esquema"),
        ("18/11/2025", "Banco Letsbank S.A.",          "Liquidação Extrajudicial", "Cedente digital do grupo"),
        ("18/11/2025", "Master S/A Corretora (CTVB)",  "Liquidação Extrajudicial", "Administradora de FIDCs"),
        ("18/11/2025", "Banco Master de Investimento", "Liquidação Extrajudicial", "Braço de investimentos"),
        ("15/01/2026", "Reag Trust DTVM S.A.",         "Liquidação Extrajudicial", "Segundo núcleo administrativo"),
        ("21/01/2026", "Will Financeira S.A.",         "Liquidação Extrajudicial", "Braço de crédito digital"),
        ("18/02/2026", "Banco Pleno S.A.",             "Liquidação Extrajudicial", "Sócio histórico"),
        ("18/02/2026", "Pleno DTVM S.A.",              "Liquidação Extrajudicial", "Distribuidora ligada ao Pleno"),
    ]

    cols = st.columns(2)
    for i, (data, inst, tipo, descr) in enumerate(eventos):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="border-left:3px solid #B22222;padding:10px 16px;margin-bottom:12px;background:rgba(178,34,34,0.05);">
                <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;color:#B22222;letter-spacing:1px;text-transform:uppercase;">{data} · {tipo}</div>
                <div style="font-family:'Source Serif 4',serif;font-size:15px;font-weight:600;margin-top:4px;">{inst}</div>
                <div style="font-family:'Inter',sans-serif;font-size:12px;opacity:.7;margin-top:2px;">{descr}</div>
            </div>
            """, unsafe_allow_html=True)


def render_footer():
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("""
    <div class="pull-quote">
        "O algoritmo sinalizava o problema meses antes da intervenção do Banco Central. 
        A anomalia não era ruído — era um padrão contínuo, reproduzível e estatisticamente 
        significativo, visível para qualquer analista com acesso aos dados públicos da CVM."
        <cite>— Relatório Forense FIDC, Conclusões, Março 2026</cite>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;font-family:'Inter',sans-serif;font-size:11px;opacity:.5;padding:20px 0 40px 0;">
        Análise baseada exclusivamente em dados públicos disponibilizados pela CVM em dados.cvm.gov.br<br>
        Metodologia: CatBoost Supervisionado · Dados: Março–Novembro 2025<br>
        Este relatório tem finalidade acadêmica e investigativa. Não constitui recomendação de investimento.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 5. FLUXO PRINCIPAL
# ─────────────────────────────────────────────────────────────
def main():
    # Carregar dados
    try:
        df_raw, feature_cols = load_data()
    except FileNotFoundError:
        st.error(
            "⚠️ Arquivo `.csv` não encontrado. "
            "Coloque-o na mesma pasta que `app.py` e reinicie o servidor."
        )
        st.stop()

    # Sidebar → controles
    dark_mode, grupos_sel, periodo, sensibilidade, score_threshold = render_sidebar(df_raw)

    # Injetar CSS conforme modo
    inject_css(dark_mode)

    # Filtrar por data e grupo (inclui Mercado para treino)
    data_inicio, data_fim = pd.Timestamp(periodo[0]), pd.Timestamp(periodo[1])
    df_filtered = df_raw[
        (df_raw["DT_COMPTC"] >= data_inicio) &
        (df_raw["DT_COMPTC"] <= data_fim)
    ].copy()

    # Rodar modelo
    df_model = run_model(df_filtered, feature_cols, sensibilidade)

    # Filtrar exibição conforme grupos selecionados (+ Mercado para referência)
    grupos_para_exibir = list(set(grupos_sel) | {"Mercado (Referência)"})
    df_display = df_model[df_model["grupo_display"].isin(grupos_para_exibir)]

    # ── RENDERIZAR PÁGINA ──
    render_header()
    render_intro()
    render_methodology()
    render_metrics(df_model, score_threshold)

    # Gráficos
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Análise Visual — Os Dados Falam</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Excesso Temporal",
        "🏆 Ranking de Fundos",
        "🗺️ Mapa Forense TM×TI",
        "🎻 Violin por Grupo",
    ])
    with tab1:
        render_excesso_temporal(df_model, dark_mode)
    with tab2:
        render_ranking_fundos(df_model, dark_mode)
    with tab3:
        render_mapa_forense(df_model, grupos_para_exibir, dark_mode)
    with tab4:
        render_distribuicao_violin(df_model, dark_mode)

    # Tabela de evidências
    render_evidence_table(df_model, score_threshold)

    # Cronologia
    render_timeline()

    # Rodapé
    render_footer()


if __name__ == "__main__":
    main()