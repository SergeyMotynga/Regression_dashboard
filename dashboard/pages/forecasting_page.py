import streamlit as st
import pandas as pd
import asyncio
import matplotlib.pyplot as plt
import numpy as np
from dashboard.data_processing.info_about_dataframe import info_about_dataframe
from dashboard.data_processing.render_main_panel import render_main_panel
from dashboard.processing.linearRegressionModel import LinearRegressionModel
from dashboard.utils.data_limiting import limit_data_to_last_points
from typing import Optional, List

def render_data_overview(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Отображает верхнюю панель с общей информацией о данных и предпросмотром DataFrame
    """
    features_size, tuples_size, = info_about_dataframe(df)
    top_cols = st.columns([4, 8])
    with top_cols[0]:
        st.markdown("#### Информация о файле:")
        st.markdown("")
        st.markdown(f"##### Кол-во записей: {tuples_size if tuples_size is not None else 'Нет информации'}")
        st.markdown(f"##### Количество признаков: {features_size if features_size is not None else 'Нет информации'}")
        st.markdown(f"##### Количество выбросов: {f'{outlier_percentage}% от всех значений' if outlier_percentage is not None else 'Нет информации'}")
    with top_cols[1]:
        st.markdown("#### Предпросмотр:")
        filtered_df = st.session_state.get('filtered_df', df)
        if filtered_df is not None:
            st.dataframe(filtered_df)
        else:
            st.markdown("Нет информации", unsafe_allow_html=True)



def render_prediction_func(df: pd.DataFrame):
    st.markdown("<h3 style='text-align: center;'>Вывод формулы</h3>", unsafe_allow_html=True)

    # Используем отфильтрованные данные
    filtered_df = st.session_state.get('filtered_df', df)
    selected_sensors = st.session_state.get('selected_sensors', df.columns.tolist())
    equation_cols = st.columns([4,4])

    with equation_cols[0]:
        # Выбор целевого параметра
        selected_aim_option = st.selectbox(
            "Выберите целевой параметр",
            options=selected_sensors,
            key="selected_aim_option"
        )

    # Получаем доступные колонки (без целевого)
    available_features = [col for col in selected_sensors if col != selected_aim_option]

    with equation_cols[1]:
        # Multiselect с фильтром
        second_options = st.multiselect(
            "Выберите признаки для построения уравнения:",
            options=available_features,
            default=st.session_state.get("selected_features", []),
            key="selected_features"  # Это ключевой момент — Streamlit сам управляет session_state
        )
    if second_options:
        model = LinearRegressionModel()
        model.fit(filtered_df[second_options], filtered_df[selected_aim_option], target_name=selected_aim_option)

         # Сохраняем модель в session_state
        st.session_state['trained_model'] = model
        st.session_state['equation_origin'] = model.get_equation(latex_output=True)


        return second_options
    else:
        st.info("Выберите хотябы один призднак для построения уравнения")

def render_equation(option):
    @st.dialog("Заполните значения для признаков")
    def equation_dialog():
        values = {}
        for op in option:
            values[op] = st.number_input(
                f"Введите значение для признака {op}: ",
                value=1.0,
                format="%f"
            )
        if st.button("Подтвердить"):
            st.session_state.equation_values = values
            st.session_state.dialog_submitted = True
            st.session_state.show_dialog = False  # Сбрасываем флаг диалога для его закрытия
            st.rerun()
    
    if st.session_state.get("show_dialog", False):
        equation_dialog()
    
    # Возвращаем значения, если они есть и диалог завершен
    if 'equation_values' in st.session_state and st.session_state.get('dialog_submitted', False):
        return st.session_state.equation_values
    return None


def render_forecasting_page(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Рендерит страницу "Прогнозирование"
    """
    st.title("Прогнозирование")
    render_data_overview(df, outlier_percentage)
    if df is not None:
        current_df_hash = hash(pd.util.hash_pandas_object(df, index=True).sum())
        if st.session_state.get('last_df_hash') != current_df_hash:
            st.session_state.clear()
            st.rerun
            st.session_state['selected_sensors'] = df.columns.tolist()
            st.session_state['sensor_editor_temp'] = df.columns.tolist()
            st.session_state['target_sensor'] = df.columns[0]
            st.session_state['last_df_hash'] = current_df_hash
            st.session_state['original_df'] = df  # Сохраняем оригинальный DataFrame
        render_main_panel(df)

    if df is not None and not df.empty:
        second_options = render_prediction_func(df)

        col1, col2 = st.columns([9, 3])
        
        values = render_equation(second_options)

        with col1:
            if st.session_state.get('equation_origin'):
                st.markdown(f"""
                    <div style="
                        display: flex;
                        align-items: center;      /* Вертикальное центрирование */
                        justify-content: center;  /* Горизонтальное центрирование */
                        font-size: 24px;
                        padding: 10px;
                        border: 3px solid #ccc;
                        width: 100%;
                        height: 100px;            /* Можно изменить под нужды */
                        box-sizing: border-box;
                        margin-bottom: 10px;
                    ">
                        {st.session_state.equation_origin}
                    </div>
                    """, unsafe_allow_html=True)
                
            sub_col1, sub_col2 = st.columns([2, 2])
            with sub_col1:
                if st.session_state.get('trained_model'):
                    st.markdown("""
                        <style>
                            .stButton button {
                                font-size: 24px !important;
                                padding: 15px 30px !important;
                                height: auto !important;
                                min-height: 50px !important;
                                width: 100% !important;
                            }
                        </style>
                        """, unsafe_allow_html=True)
                    
                    input_btn = st.button("Ввести значения для формулы", key="open_dialog_button")
                    if input_btn and second_options is not None:
                        st.session_state.show_dialog = True
                        st.session_state.dialog_submitted = False
                        st.rerun()

            with sub_col2:
                with sub_col2:
                    selected_features = st.session_state.get("selected_features", [])
                    if values is not None and selected_features:
                        try:
                            # Проверяем, что есть выбранные признаки
                            if not selected_features:
                                st.warning("Не выбрано ни одного признака для предсказания.")
                                return

                            values_list = [values[feature] for feature in selected_features]

                            # Проверяем, что все признаки были переданы
                            if len(values_list) == 0:
                                st.warning("Нет данных для указанных признаков.")
                                return

                            model = st.session_state['trained_model']
                            prediction = model.predict([values_list])[0]

                            st.markdown(f"""
                                <div style="
                                    display: flex;
                                    align-items: center;
                                    justify-content: left;
                                    font-size: 20px;
                                    padding: 10px;
                                    width: 100%;
                                    height: 100px;">
                                    Итог точечного предсказания: {prediction:.2f}
                                </div>
                            """, unsafe_allow_html=True)

                        except Exception as e:
                            st.warning("Вы не выбрали параметры")

        with col2:
            if st.session_state.get('trained_model'):
                model = st.session_state['trained_model']
                metrics = model.evaluate()

                # Отображение метрик с пояснениями
                st.markdown("### Метрики качества")

                # R² Score с цветовой индикацией
                r2 = metrics['R2']
                if r2 >= 0.9:
                    r2_color = "green"
                    r2_interpretation = "Отлично"
                elif r2 >= 0.7:
                    r2_color = "orange"
                    r2_interpretation = "Хорошо"
                elif r2 >= 0.5:
                    r2_color = "orange"
                    r2_interpretation = "Средне"
                else:
                    r2_color = "red"
                    r2_interpretation = "Плохо"

                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: rgba(128, 128, 128, 0.1);">
                    <h4 style="margin: 0;">R² (Коэффициент детерминации)</h4>
                    <p style="font-size: 28px; margin: 5px 0; color: {r2_color};"><b>{r2:.4f}</b></p>
                    <p style="margin: 0; font-size: 14px;">Качество: <b>{r2_interpretation}</b></p>
                    <p style="margin: 5px 0; font-size: 12px; color: gray;">
                        Показывает долю дисперсии целевой переменной, объясняемую моделью.
                        Чем ближе к 1, тем лучше модель описывает данные.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Остальные метрики
                metrics_df = pd.DataFrame([
                    {"Метрика": "MAE", "Значение": f"{metrics['MAE']:.4f}", "Описание": "Средняя абсолютная ошибка"},
                    {"Метрика": "MSE", "Значение": f"{metrics['MSE']:.4f}", "Описание": "Средняя квадратичная ошибка"},
                    {"Метрика": "RMSE", "Значение": f"{metrics['RMSE']:.4f}", "Описание": "Корень из MSE"}
                ])

                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Графики по центру друг под другом
        if st.session_state.get('trained_model'):
            model = st.session_state['trained_model']
            selected_features = st.session_state.get("selected_features", [])
            selected_aim = st.session_state.get("selected_aim_option")
            filtered_df = st.session_state.get('filtered_df', df)

            if selected_features and selected_aim:
                X = filtered_df[selected_features]
                y_true = filtered_df[selected_aim].values
                y_pred = model.predict(X)

                # Первый график
                st.markdown("### Реальные vs Предсказанные значения")

                fig1, ax1 = plt.subplots(figsize=(12, 5))

                # Строим линии реальных и предсказанных значений
                indices = np.arange(len(y_true))
                ax1.plot(indices, y_true, label='Реальные значения', color='blue', linewidth=2, alpha=0.7)
                ax1.plot(indices, y_pred, label='Предсказанные формулой', color='red', linewidth=2, alpha=0.7, linestyle='--')

                ax1.set_xlabel('Номер записи', fontsize=11)
                ax1.set_ylabel(f'{selected_aim}', fontsize=11)
                ax1.set_title('Реальные значения vs Предсказания модели', fontsize=13)
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)

                st.pyplot(fig1)
                plt.close()

                st.markdown("""
                <p style="font-size: 12px; color: gray; text-align: center; margin-bottom: 30px;">
                    Чем ближе красная линия к синей, тем точнее формула описывает целевой параметр
                </p>
                """, unsafe_allow_html=True)

                # Второй график
                st.markdown("### Качество предсказаний (R²)")

                fig2, ax2 = plt.subplots(figsize=(12, 5))

                # Scatter plot: реальные vs предсказанные
                ax2.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5, color='blue', s=50)

                # Линия идеального предсказания y=x
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное предсказание')

                ax2.set_xlabel('Реальные значения', fontsize=11)
                ax2.set_ylabel('Предсказанные значения', fontsize=11)
                ax2.set_title('Реальные vs Предсказанные значения', fontsize=13)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig2)
                plt.close()

                st.markdown("""
                <p style="font-size: 12px; color: gray; text-align: center;">
                    Чем ближе точки к красной линии, тем выше R² и точнее модель
                </p>
                """, unsafe_allow_html=True)