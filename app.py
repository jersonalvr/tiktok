import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import json

# 🚀 Datos Fijos
COINS_PER_DOLLAR = 70  # 70 monedas cuestan $1 en TikTok (dato inamovible)
COINS_FOR_CREATOR_DOLLAR = 200  # El creador recibe $1 cada 200 monedas (dato inamovible)

# 🎁 Tabla de Regalos en TikTok con Bootstrap Icons
GIFT_MENU = {
    'Rosa': {'coins': 1, 'icon': '🌹'},
    'TikTok': {'coins': 1, 'icon': '📱'},
    'Perfume': {'coins': 20, 'icon': '💐'},
    'Avión de Papel': {'coins': 99, 'icon': '✈️'},
    'Anillo de Diamantes': {'coins': 300, 'icon': '💍'},
    'Coche de Carreras': {'coins': 7000, 'icon': '🏎️'},
    'León': {'coins': 29999, 'icon': '🦁'},
    'Castillo': {'coins': 20000, 'icon': '🏰'},
    'Yate': {'coins': 98888, 'icon': '🛥️'},
    'Planeta': {'coins': 15000, 'icon': '🪐'}
}

# 📈 Engagement Ratio Predeterminado
DEFAULT_ENGAGEMENT_RATIO = 0.005  # 0.5%

# 🧰 Funciones Auxiliares
rapidapi_key = st.secrets["RAPIDAPI"]["key"]

def generar_respuesta_rapidapi(prompt):
    """
    Función para generar respuestas utilizando la API de RapidAPI.
    """
    url = "https://cheapest-gpt-4-turbo-gpt-4-vision-chatgpt-openai-ai-api.p.rapidapi.com/v1/chat/completions"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "gpt-4o",
        "max_tokens": 1000,
        "temperature": 0.7
    }
    headers = {
        "x-rapidapi-key": rapidapi_key,  # Reemplaza con tu clave de RapidAPI
        "x-rapidapi-host": "cheapest-gpt-4-turbo-gpt-4-vision-chatgpt-openai-ai-api.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except requests.RequestException as e:
        return f"Error al comunicarse con la API de RapidAPI: {e}"
    except Exception as e:
        return f"Error inesperado: {e}"

@st.cache_data
def analizar_donaciones(total_donation, selected_gifts, engagement_ratio):
    """
    Función principal para analizar las donaciones.
    """
    # Cálculos básicos
    total_coins = total_donation * COINS_FOR_CREATOR_DOLLAR
    viewers_spending = total_coins / COINS_PER_DOLLAR
    tiktok_gross_income = viewers_spending - total_donation

    # Cálculos para cada regalo
    gift_df = pd.DataFrame.from_dict(GIFT_MENU, orient='index')
    gift_df.reset_index(inplace=True)
    gift_df.rename(columns={'index': 'Regalo'}, inplace=True)
    gift_df['Costo por Regalo ($)'] = gift_df['coins'] / COINS_PER_DOLLAR
    gift_df['Número de Regalos Necesarios'] = (total_coins / gift_df['coins']).astype(int)
    gift_df['Número de Espectadores Necesarios'] = (gift_df['Número de Regalos Necesarios'] / engagement_ratio).astype(int)

    # Filtrar los regalos seleccionados
    if selected_gifts:
        gift_df = gift_df[gift_df['Regalo'].isin(selected_gifts)]

    # Estimación detallada de gastos de TikTok
    platform_fee_rate = 0.15  # 15% de comisiones de plataforma
    google_play_fee = viewers_spending * platform_fee_rate * 0.5  # 50% de transacciones vía Google Play
    app_store_fee = viewers_spending * platform_fee_rate * 0.5  # 50% de transacciones vía App Store
    server_costs = tiktok_gross_income * 0.1  # 10% de los ingresos brutos
    personnel_costs = tiktok_gross_income * 0.15  # 15% de los ingresos brutos
    marketing_costs = tiktok_gross_income * 0.05  # 5% de los ingresos brutos para marketing
    r_and_d_costs = tiktok_gross_income * 0.1  # 10% de los ingresos brutos para R&D
    other_costs = tiktok_gross_income * 0.05  # 5% para otros gastos imprevistos

    total_expenses = (google_play_fee + app_store_fee + server_costs + personnel_costs +
                      marketing_costs + r_and_d_costs + other_costs)
    tiktok_net_income = tiktok_gross_income - total_expenses

    # Popularidad del creador
    num_viewers = gift_df['Número de Espectadores Necesarios'].min()
    total_users_needed = num_viewers / engagement_ratio

    # Preparación de datos para visualizaciones
    finance_data = {
        'Creador': total_donation,
        'TikTok (Neto)': tiktok_net_income,
        'Google Play': google_play_fee,
        'App Store': app_store_fee,
        'Gastos Operativos': server_costs + personnel_costs + marketing_costs + r_and_d_costs + other_costs
    }

    tiktok_finances = pd.DataFrame({
        'Categoría': ['Servidores', 'Personal', 'Comisiones Google Play', 'Comisiones App Store',
                      'Marketing', 'R&D', 'Otros Gastos', 'Ingreso Neto'],
        'Monto': [server_costs, personnel_costs, google_play_fee, app_store_fee,
                  marketing_costs, r_and_d_costs, other_costs, tiktok_net_income]
    })

    # Gráfico de pastel
    fig_finance = px.pie(
        names=list(finance_data.keys()),
        values=list(finance_data.values()),
        title="Distribución de Ingresos y Gastos 💰"
    )

    # Gráfico de barras
    fig_finances = px.bar(tiktok_finances, x='Categoría', y='Monto',
                          title="Desglose de Gastos e Ingresos de TikTok 📊")

    # Simulación de carga del servidor
    def simulate_server_load(total_users, duration_hours):
        time_points = np.linspace(0, duration_hours, num=100)
        base_load = total_users * np.sin(np.pi * time_points / duration_hours) + total_users
        noise = np.random.normal(0, total_users * 0.1, 100)
        return time_points, base_load + noise

    event_duration_hours = 4  # Duración del evento
    time_points, server_load = simulate_server_load(total_users_needed, event_duration_hours)

    fig_server_load = px.line(x=time_points, y=server_load,
                              labels={'x': 'Tiempo (horas)', 'y': 'Número de usuarios activos'},
                              title='Simulación de Carga del Servidor Durante el Evento 🚀')

    # Simulación de transacciones en tiempo real
    transaction_data = []
    for i in range(1000):
        new_transaction = {
            'timestamp': datetime.now() + timedelta(seconds=i * 30),
            'gift': np.random.choice(gift_df['Regalo'], p=[1/len(gift_df)]*len(gift_df)),
            'user_id': f"user_{np.random.randint(1, int(total_users_needed))}"
        }
        new_transaction['coins'] = GIFT_MENU[new_transaction['gift']]['coins']
        transaction_data.append(new_transaction)
    df_transactions = pd.DataFrame(transaction_data)

    fig_transactions = px.scatter(df_transactions, x='timestamp', y='coins', color='gift',
                                  title="Simulación de Transacciones en Tiempo Real 🕒",
                                  labels={'coins': 'Monedas'})

    # Preparación de métricas clave
    metrics = {
        "Gasto Total": f"${total_expenses:,.2f}",
        "Gasto Total de Espectadores": f"${viewers_spending:,.2f}",
        "Número de Espectadores Necesarios": f"{num_viewers:,.0f}",
        "Número Total de Usuarios Necesarios": f"{int(total_users_needed):,}",
        "Ingreso Neto de TikTok": f"${tiktok_net_income:,.2f}",
        "Tasa de Retención de TikTok": f"{(tiktok_net_income/viewers_spending)*100:.2f}%"
    }

    # Serializar GIFT_MENU para evitar errores de formato en el prompt
    serialized_gift_menu = json.dumps(GIFT_MENU, ensure_ascii=False, indent=4)

    # Generar análisis detallado utilizando la función de RapidAPI
    prompt = f"""
    Realiza un análisis técnico detallado sobre los datos financieros y técnicos proporcionados a continuación.
    Incluye aspectos como arquitectura de sistemas, tecnologías utilizadas, estimaciones de gastos y conclusiones.
    Asegúrate de presentar las fórmulas y cálculos en formato Markdown para una correcta visualización.

    **Datos Fijos:**
    - Monedas por dólar: {COINS_PER_DOLLAR}
    - Monedas para que el creador reciba $1: {COINS_FOR_CREATOR_DOLLAR}

    **Datos Financieros:**
    - Total de Monedas: {total_coins}
    - Gasto Total de Espectadores: ${viewers_spending:,.2f}
    - Número de Espectadores que donaron: {num_viewers:,.0f}
    - Número Total de Usuarios Necesarios: {int(total_users_needed):,}
    - Ingreso Neto de TikTok: ${tiktok_net_income:,.2f}

    **Desglose de Gastos de TikTok:**
    - Servidores: ${server_costs:,.2f}
    - Personal: ${personnel_costs:,.2f}
    - Comisiones Google Play: ${google_play_fee:,.2f}
    - Comisiones App Store: ${app_store_fee:,.2f}
    - Marketing: ${marketing_costs:,.2f}
    - R&D: ${r_and_d_costs:,.2f}
    - Otros Gastos: ${other_costs:,.2f}

    **Aspectos Técnicos:**
    - Duración del Evento: {event_duration_hours} horas
    - Simulación de Carga del Servidor: Datos adjuntos
    - Menú de Regalos: {serialized_gift_menu}
    """
    analysis = generar_respuesta_rapidapi(prompt)

    # Preparación del análisis de monetización
    monetization_analysis = (
        r"""
        ### Análisis de Monetización

        Monedas por Dólar:
        $$\text{Monedas por Dólar} = """ + f"{COINS_PER_DOLLAR}" + r"""$$

        Monedas para que el creador reciba $1:
        $$\text{Monedas para }\$1\text{ al creador} = """ + f"{COINS_FOR_CREATOR_DOLLAR}" + r"""$$

        Ingreso por Moneda:
        $$\text{Ingreso por Moneda} = \frac{1}{""" + f"{COINS_PER_DOLLAR}" + r"""} - \frac{1}{""" + f"{COINS_FOR_CREATOR_DOLLAR}" + r"""}$$

        Ingreso Neto de TikTok:
        $$\text{Ingreso Neto} = ${tiktok_gross_income:,.2f}$$

        Gasto Total de Espectadores:
        $$\text{Monedas gastadas} = \text{Total de Monedas} = {total_coins:,.0f}$$
        $$\text{Gasto Total de Espectadores} = ${viewers_spending:,.2f}$$
        """
    )

    # Devolver todos los elementos para su visualización
    return (fig_finance, fig_finances, fig_server_load, fig_transactions, metrics,
            analysis, gift_df, monetization_analysis, num_viewers, total_users_needed)

def chatbot_response(user_input):
    """
    Función para obtener la respuesta del chatbot.
    """
    response = generar_respuesta_rapidapi(user_input)
    return response

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Análisis Técnico Donaciones TikTok 📱",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos para Bootstrap Icons
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
    <style>
        .bi {
            font-size: 1.5rem;
        }
        .gift-icon {
            font-size: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Título y descripción
st.title("Análisis Técnico Extendido de Donaciones en TikTok 📱")
st.markdown("Este dashboard ofrece un análisis profundo de las donaciones en TikTok, incluyendo aspectos técnicos y financieros. ¡Vamos a sumergirnos! 🌊")

# Sidebar para la navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a", ["Análisis Financiero 💰", "Métricas Clave 📈", "Análisis Detallado 📝",
                                 "Análisis de Monetización 💡", "Análisis Técnico Detallado ⚙️", "Chatbot 🤖"])

# Parámetros de entrada en la barra lateral
st.sidebar.header("Parámetros de Análisis")
total_donation = st.sidebar.slider("Donación total al creador ($)", min_value=1000, max_value=1000000, step=1000, value=500000)
selected_gifts = st.sidebar.multiselect("Selecciona los Regalos 🎁", options=list(GIFT_MENU.keys()), default=['León'])

# Menú deslizante para Ratio de engagement
engagement_ratio = st.sidebar.slider("Ratio de Engagement (%)", min_value=0.1, max_value=10.0, step=0.1, value=DEFAULT_ENGAGEMENT_RATIO * 100) / 100

# Función para ejecutar el análisis en un hilo separado
def run_analysis():
    with st.spinner('Analizando...'):
        outputs = analizar_donaciones(total_donation, selected_gifts, engagement_ratio)
        (fig_finance, fig_finances, fig_server_load, fig_transactions, metrics,
         analysis, gift_df, monetization_analysis, num_viewers, total_users_needed) = outputs

        # Almacenar resultados en el estado de la sesión
        st.session_state['fig_finance'] = fig_finance
        st.session_state['fig_finances'] = fig_finances
        st.session_state['fig_server_load'] = fig_server_load
        st.session_state['fig_transactions'] = fig_transactions
        st.session_state['metrics'] = metrics
        st.session_state['analysis'] = analysis
        st.session_state['gift_df'] = gift_df
        st.session_state['monetization_analysis'] = monetization_analysis
        st.session_state['num_viewers'] = num_viewers
        st.session_state['total_users_needed'] = total_users_needed
        st.session_state['analysis_done'] = True

# Botón para ejecutar el análisis con barra de progreso
if st.sidebar.button("Analizar 🔍"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Ejecutar el análisis con simulación de progreso
    for i in range(100):
        time.sleep(0.05)  # Simulamos un proceso que toma tiempo
        progress_bar.progress(i + 1)
        status_text.text(f"Análisis en progreso: {i+1}%")

    # Realizar el análisis real
    run_analysis()

    # Limpiar la barra de progreso y el texto de estado
    progress_bar.empty()
    status_text.empty()

    st.success("¡Análisis completado!")

# Mostrar contenido según la página seleccionada
if page == "Análisis Financiero 💰":
    st.header("Análisis Financiero 💰")
    if st.session_state.get('analysis_done', False):
        st.subheader("Menú de Regalos y Cálculos 📜")
        # Mostrar regalos con íconos
        for _, row in st.session_state['gift_df'].iterrows():
            st.markdown(f"**{row['Regalo']}** {GIFT_MENU[row['Regalo']]['icon']}:", unsafe_allow_html=True)
            st.markdown(f"- **Costo por Regalo ($)**: {row['Costo por Regalo ($)']:.2f}")
            st.markdown(f"- **Número de Regalos Necesarios**: {row['Número de Regalos Necesarios']}")
            st.markdown(f"- **Número de Espectadores Necesarios**: {row['Número de Espectadores Necesarios']}")
            st.markdown("---")
        
        st.plotly_chart(st.session_state['fig_finance'], use_container_width=True)
        st.plotly_chart(st.session_state['fig_finances'], use_container_width=True)
        st.plotly_chart(st.session_state['fig_server_load'], use_container_width=True)
        st.plotly_chart(st.session_state['fig_transactions'], use_container_width=True)
    else:
        st.info("Por favor, ingresa los parámetros y haz clic en 'Analizar' en la barra lateral.")

elif page == "Métricas Clave 📈":
    st.header("Métricas Clave 📈")
    if st.session_state.get('analysis_done', False):
        gasto_total_formula = r'''\text{Gasto Total} = \text{Servidores} + \text{Personal} + \text{Comisiones Google Play} + \text{Comisiones App Store} + \text{Marketing} + \text{R\&D} + \text{Otros Gastos}'''
        # Mostrar la fórmula de Gasto Total
        st.subheader("Fórmula de Gasto Total")
        st.latex(gasto_total_formula)
        metrics = st.session_state['metrics']
        for key, value in metrics.items():
            st.subheader(f"{key}")
            st.markdown(value)
    else:
        st.info("Por favor, ingresa los parámetros y haz clic en 'Analizar' en la barra lateral.")

elif page == "Análisis Detallado 📝":
    st.header("Análisis Detallado 📝")
        
    # Inyectar CSS personalizado
    st.markdown(
        r"""
        <style>
            .katex-html {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True
    )        
    if st.session_state.get('analysis_done', False):
        analysis = st.session_state['analysis']
        st.markdown(analysis, unsafe_allow_html=True)  
    else:
        st.info("Por favor, ingresa los parámetros y haz clic en 'Analizar' en la barra lateral.")

elif page == "Análisis de Monetización 💡":
    st.header("Análisis de Monetización 💡")
    if st.session_state.get('analysis_done', False):
        total_coins = total_donation * COINS_FOR_CREATOR_DOLLAR
        viewers_spending = total_coins / COINS_PER_DOLLAR
        tiktok_gross_income = viewers_spending - total_donation
        
        st.subheader("Análisis de Monetización")

        st.write("Monedas por Dólar:")
        st.latex(r"\text{Monedas por Dólar} = " + str(COINS_PER_DOLLAR))

        st.write("Monedas para que el creador reciba $1:")
        st.latex(r"\text{Monedas para }\$1\text{ al creador} = " + str(COINS_FOR_CREATOR_DOLLAR))

        st.write("Ingreso por Moneda:")
        st.latex(r"\text{Ingreso por Moneda} = \frac{1}{" + str(COINS_PER_DOLLAR) + r"} - \frac{1}{" + str(COINS_FOR_CREATOR_DOLLAR) + r"}")

        st.write("Ingreso Neto de TikTok:")
        st.latex(r"\text{Ingreso Neto} = \$" + f"{tiktok_gross_income:.2f}")

        st.write("Gasto Total de Espectadores:")
        st.latex(r"\text{Monedas gastadas} = \text{Total de Monedas} = " + f"{total_coins:,.0f}")
        st.latex(r"\text{Gasto Total de Espectadores} = \$" + f"{viewers_spending:.2f}")
    else:
        st.info("Por favor, ingresa los parámetros y haz clic en 'Analizar' en la barra lateral.")

elif page == "Análisis Técnico Detallado ⚙️":
    st.header("Análisis Técnico Detallado ⚙️")
    if st.session_state.get('analysis_done', False):
        num_viewers = st.session_state['num_viewers']
        total_users_needed = st.session_state['total_users_needed']
        engagement_ratio = engagement_ratio  # Usar el valor seleccionado

        technical_content = f"""
        ## Arquitectura de Microservicios 🏗️

        TikTok utiliza una arquitectura de microservicios para manejar su vasto ecosistema. Esto permite una escalabilidad horizontal y un despliegue más ágil. Cada servicio es independiente y puede ser desarrollado y escalado por separado.

        **Ejemplo de servicios:**

        - **Servicio de Usuarios**: Maneja autenticación, perfiles y gestión de usuarios.
        - **Servicio de Donaciones**: Procesa transacciones y donaciones en tiempo real.
        - **Servicio de Streaming**: Gestiona la transmisión de video en vivo.
        - **Servicio de Notificaciones**: Envía actualizaciones y alertas a los usuarios.

        ## Bases de Datos Distribuidas 🗄️

        Para manejar la gran cantidad de datos generados, TikTok emplea bases de datos distribuidas como Cassandra o MongoDB.

        **Ejemplo de esquema en MongoDB:**

        ```json
        {{
            "donation_id": "donation_12345",
            "user_id": "user_6789",
            "creator_id": "creator_1011",
            "gift": "León",
            "coins": 29999,
            "timestamp": "2023-10-01T12:34:56Z"
        }}
        ```

        ## Procesamiento en Tiempo Real ⏱️

        Utilizan Apache Kafka para manejar streams de datos en tiempo real, permitiendo que los datos de donaciones y comentarios se procesen al instante.

        **Ejemplo de productor Kafka en Python:**

        ```python
        from kafka import KafkaProducer
        import json

        producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                                 value_serializer=lambda v: json.dumps(v).encode('utf-8'))

        donation_event = {{
            "donation_id": "donation_12345",
            "user_id": "user_6789",
            "gift": "León",
            "coins": 29999
        }}

        producer.send('donations_topic', donation_event)
        ```

        ## Caching con Redis ⚡

        Para reducir la latencia y descargar las bases de datos principales, se utiliza Redis como sistema de caché.

        **Ejemplo de almacenamiento en caché:**

        ```python
        import redis

        r = redis.Redis(host='localhost', port=6379, db=0)

        # Almacenar el total de donaciones para un creador
        r.set('creator_1011_total_donations', 500000)

        # Obtener el valor almacenado
        total_donations = r.get('creator_1011_total_donations')
        ```

        ## Infraestructura en la Nube ☁️

        TikTok despliega sus servicios en plataformas en la nube como AWS o GCP, utilizando contenedores Docker y orquestación con Kubernetes.

        **Ejemplo de archivo Dockerfile para el servicio de donaciones:**

        ```dockerfile
        FROM python:3.9-slim

        WORKDIR /app

        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt

        COPY . .
        CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
        ```

        ## Monitoreo y Alertas 🚨

        Utilizan Prometheus para recopilar métricas y Grafana para visualizar el estado de los sistemas.

        **Ejemplo de métrica personalizada:**

        ```python
        from prometheus_client import Counter, start_http_server

        donations_counter = Counter('donations_total', 'Total de donaciones recibidas')

        def process_donation(amount):
            donations_counter.inc()
            # Procesar donación
        ```

        ## Seguridad y Cumplimiento 🔒

        La seguridad es crítica, implementando prácticas como:

        - **Encriptación de datos**: Uso de TLS para datos en tránsito y AES para datos en reposo.
        - **Autenticación y Autorización**: Implementación de OAuth 2.0 y JWT para tokens de acceso.
        - **Cumplimiento Normativo**: Alineación con GDPR, CCPA y otras regulaciones de privacidad.

        ## Inteligencia Artificial y Machine Learning 🤖

        TikTok utiliza modelos de ML para:

        - **Recomendación de Contenido**: Algoritmos que personalizan el feed de cada usuario.
        - **Detección de Fraude**: Sistemas que identifican actividades sospechosas en tiempo real.
        - **Moderación de Contenido**: Uso de visión por computadora para detectar contenido inapropiado.

        **Ejemplo de modelo de detección de anomalías:**

        ```python
        from sklearn.ensemble import IsolationForest

        # Datos de transacciones
        X = [[coins] for coins in df_transactions['coins']]

        # Entrenar el modelo
        clf = IsolationForest(random_state=0).fit(X)

        # Predecir anomalías
        anomalies = clf.predict(X)
        df_transactions['anomaly'] = anomalies
        ```

        ## Estimaciones de Infraestructura 📊

        - **Número de Servidores**: Basado en la carga simulada, se estima que se requieren al menos 1000 servidores para manejar el tráfico durante el evento.
        - **Ancho de Banda**: Se estima un uso de ancho de banda de 2 Tbps durante los picos del evento.
        - **Almacenamiento**: Con millones de transacciones, se necesitan sistemas de almacenamiento escalables, como Amazon S3 o Google Cloud Storage.

        ## Popularidad del Creador 🌟

        Para lograr que {num_viewers:,.0f} espectadores (donantes) contribuyan con un León (29,999 monedas), el creador debe tener una base de seguidores muy grande. Asumiendo una tasa de conversión del {engagement_ratio * 100:.2f}%, necesitaría al menos {int(total_users_needed):,} seguidores activos.
        """
        st.markdown(technical_content)

    else:
        st.info("Por favor, ingresa los parámetros y haz clic en 'Analizar' en la barra lateral.")

elif page == "Chatbot 🤖":
    st.header("Chatbot 🤖")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'suggested_messages' not in st.session_state:
        st.session_state['suggested_messages'] = [
            "¿Cómo puedo aumentar mis donaciones?",
            "¿Qué regalos son más populares?",
            "¿Cómo afecta el engagement a mis ingresos?",
            "Explícame el análisis financiero.",
            "¿Cómo funciona el chatbot?"
        ]

    st.write("Bienvenido al Chatbot. Selecciona un mensaje sugerido o escribe tu propio mensaje a continuación.")

    # Mostrar mensajes sugeridos
    st.subheader("Mensajes Sugeridos")
    for msg in st.session_state['suggested_messages']:
        if st.button(msg):
            user_input = msg
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            bot_response = chatbot_response(user_input)
            st.session_state['chat_history'].append({"role": "assistant", "content": bot_response})

    st.markdown("---")

    # Entrada de usuario
    user_input = st.text_input("Escribe tu mensaje aquí...")

    if st.button("Enviar 🚀"):
        if user_input:
            # Append user input to chat history
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            # Get bot response
            bot_response = chatbot_response(user_input)
            st.session_state['chat_history'].append({"role": "assistant", "content": bot_response})
            # Limpiar la entrada de usuario
            user_input = ''

    # Mostrar historial de chat
    st.subheader("Historial de Chat")
    for message in st.session_state['chat_history']:
        if message['role'] == 'user':
            st.markdown(f"**Tú:** {message['content']}")
        else:
            st.markdown(f"**Chatbot:** {message['content']}")

elif page == "Análisis de Monetización 💡":
    st.header("Análisis de Monetización 💡")
    if st.session_state.get('analysis_done', False):
        monetization_analysis = st.session_state['monetization_analysis']
        st.markdown(monetization_analysis)
    else:
        st.info("Por favor, ingresa los parámetros y haz clic en 'Analizar' en la barra lateral.")

# Asegurarse de que el estado de la sesión esté inicializado
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'suggested_messages' not in st.session_state:
    st.session_state['suggested_messages'] = [
        "¿Cómo puedo aumentar mis donaciones?",
        "¿Qué regalos son más populares?",
        "¿Cómo afecta el engagement a mis ingresos?",
        "Explícame el análisis financiero.",
        "¿Cómo funciona el chatbot?"
    ]

# Footer con año dinámico
current_year = datetime.now().year
st.markdown(f"""
    <hr>
    <div style="text-align: center;">
        <p>Desarrollado por Jerson Ruiz 👨‍💻 | © {current_year}</p>
        <p>
            <a href="https://github.com/jersonalvr" target="_blank" style="margin: 0 15px;">
                <i class="bi bi-github"></i>
            </a>
            <a href="https://linkedin.com/in/jersonalvr" target="_blank" style="margin: 0 15px;">
                <i class="bi bi-linkedin"></i>
            </a>
            <a href="https://twitter.com/jersonalvr" target="_blank" style="margin: 0 15px;">
                <i class="bi bi-twitter"></i>
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)
