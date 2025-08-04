import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Annotation Tool",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/spans_df_20250716.csv')
    return df


def load_annotations():
    if os.path.exists('data/annotations/trace_annotations.json'):
        with open('data/annotations/trace_annotations.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_annotations(annotations):
    with open('data/annotations/trace_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


def build_tree(df):
    nodes = {}
    for _, row in df.iterrows():
        node_id = row['context.span_id']
        nodes[node_id] = {
            'id': node_id,
            'name': row['name'],
            'span_kind': row['span_kind'],
            'parent_id': row['parent_id'],
            'children': [],
            'data': row.to_dict()
        }

    root_nodes = []
    for node_id, node in nodes.items():
        parent_id = node['parent_id']
        if pd.isna(parent_id) or parent_id not in nodes:
            root_nodes.append(node)
        else:
            nodes[parent_id]['children'].append(node)

    return root_nodes, nodes


def get_default_attributes(node):
    """Возвращает атрибуты по умолчанию в зависимости от типа ноды"""
    default_attrs = ['name']

    span_kind = node['span_kind']

    if span_kind == "LLM":
        llm_attrs = [
            'attributes.llm.input_messages',
            'attributes.llm.output_messages'
        ]
        for attr in llm_attrs:
            default_attrs.append(attr)

    elif span_kind == "RETRIEVER":
        retrieval_attrs = [
            'attributes.retrieval.documents',
            'attributes.input.value'
        ]
        for attr in retrieval_attrs:
            default_attrs.append(attr)

    else:
        general_attrs = [
            'attributes.input.value',
            'attributes.output.value'
        ]
        for attr in general_attrs:
            default_attrs.append(attr)

    return default_attrs


def display_node_with_annotation(node, annotations, depth=0, node_path=""):
    node_id = node['id']

    if node_id not in annotations:
        default_attrs = get_default_attributes(node)
        annotations[node_id] = {
            'comment': '',
            'approved': None,
            'selected_attributes': default_attrs,
            'timestamp': datetime.now().isoformat()
        }

    indent = "&nbsp;" * (depth * 4)

    # Ключи для состояния сворачивания (по умолчанию свернуто)
    node_collapsed_key = f"node_collapsed_{node_path}_{node_id}"
    if node_collapsed_key not in st.session_state:
        st.session_state[node_collapsed_key] = True
    
    with st.container():
        # Заголовок с кнопкой сворачивания
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"{indent}### 🌿 {node['name']} ({node['span_kind']})", unsafe_allow_html=True)
        with col2:
            if st.button("📦" if st.session_state[node_collapsed_key] else "📂", 
                        key=f"collapse_{node_id}", help="Свернуть/развернуть ноду"):
                st.session_state[node_collapsed_key] = not st.session_state[node_collapsed_key]
                st.rerun()
        
        # Инициализируем selected_attrs
        selected_attrs = annotations[node_id].get('selected_attributes', get_default_attributes(node))
        
        if not st.session_state[node_collapsed_key]:
            st.write(f"**ID:** {node_id}")

            all_attributes = list(node['data'].keys())
            if ('selected_attributes' not in annotations[node_id] or 
                not annotations[node_id]['selected_attributes']):
                default_attrs = get_default_attributes(node)
                annotations[node_id]['selected_attributes'] = default_attrs
                selected_attrs = default_attrs
            
            selected_attrs = st.multiselect(
                f"Атрибуты для {node['name']}:",
                all_attributes,
                default=selected_attrs,
                key=f"attrs_{node_id}",
                on_change=lambda: save_annotations(annotations)
            )
        
        # Проверяем изменения только если нода не свернута
        if not st.session_state[node_collapsed_key]:
            if selected_attrs != annotations[node_id].get('selected_attributes', []):
                annotations[node_id]['selected_attributes'] = selected_attrs
                annotations[node_id]['timestamp'] = datetime.now().isoformat()
                save_annotations(annotations)

        if not st.session_state[node_collapsed_key]:
            if selected_attrs:
                with st.expander("📋 Атрибуты", expanded=True):
                    for attr in selected_attrs:
                        if attr in node['data'] and pd.notna(node['data'][attr]):
                            value = node['data'][attr]
                            if isinstance(value, str) and len(value) > 200:
                                with st.expander(f"**{attr}:** {value[:100]}..."):
                                    st.text(value)
                            else:
                                st.write(f"**{attr}:** {value}")
                        else:
                            st.write(f"**{attr}:** (пусто)")

        # Инициализируем comment
        comment = annotations[node_id]['comment']
        
        if not st.session_state[node_collapsed_key]:
            with st.expander("💬 Комментарий", expanded=False):
                comment = st.text_area(
                    f"Комментарий к {node['name']}:",
                    value=comment,
                    key=f"comment_{node_id}",
                    height=100,
                    on_change=lambda: save_annotations(annotations)
                )
            
            if comment != annotations[node_id]['comment']:
                annotations[node_id]['comment'] = comment
                annotations[node_id]['timestamp'] = datetime.now().isoformat()
                save_annotations(annotations)

        if not st.session_state[node_collapsed_key]:
            with st.expander("🎯 Действия и статус", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ Одобрить", key=f"yes_{node_id}"):
                        annotations[node_id]['approved'] = True
                        annotations[node_id]['timestamp'] = datetime.now().isoformat()
                        save_annotations(annotations)
                        st.success("Нода одобрена!")
                        st.rerun()

                with col2:
                    if st.button("❌ Отклонить", key=f"no_{node_id}"):
                        annotations[node_id]['approved'] = False
                        annotations[node_id]['timestamp'] = datetime.now().isoformat()
                        save_annotations(annotations)
                        st.error("Нода отклонена!")
                        st.rerun()

                with col3:
                    if st.button("🔄 Сброс", key=f"clear_{node_id}"):
                        annotations[node_id]['approved'] = None
                        annotations[node_id]['timestamp'] = datetime.now().isoformat()
                        save_annotations(annotations)
                        st.info("Статус сброшен!")
                        st.rerun()

                status = annotations[node_id]['approved']
                if status is True:
                    st.success("✅ Статус: Одобрено")
                elif status is False:
                    st.error("❌ Статус: Отклонено")
                else:
                    st.info("⏳ Статус: Не размечалось")

        if node['children']:
            children_expanded_key = f"children_expanded_{node_path}_{node_id}"
            if children_expanded_key not in st.session_state:
                st.session_state[children_expanded_key] = False  # Дочерние ноды по умолчанию свернуты

            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"🌿 Дочерние ноды ({len(node['children'])})", 
                            key=f"expand_{node_id}"):
                    st.session_state[children_expanded_key] = not st.session_state[children_expanded_key]
                    st.rerun()
            with col2:
                if st.button("📦 Все", key=f"collapse_all_{node_id}", 
                            help="Свернуть все дочерние ноды"):
                    # Сворачиваем все дочерние ноды
                    for child in node['children']:
                        child_path = f"{node_path}_{node_id}" if node_path else node_id
                        child_collapsed_key = f"node_collapsed_{child_path}_{child['id']}"
                        st.session_state[child_collapsed_key] = True
                    st.rerun()

            if st.session_state[children_expanded_key]:
                with st.expander("📂 Дочерние ноды", expanded=True):
                    for i, child in enumerate(node['children']):
                        child_path = f"{node_path}_{node_id}" if node_path else node_id
                        st.write(f"**{i+1}. Дочерняя нода:**")
                        display_node_with_annotation(child, annotations, depth + 1, child_path)

        st.divider()


def display_root_node_with_children(root_node, all_nodes, annotations):
    st.header(f"🌳 Корневая нода: {root_node['name']}")
    st.write(f"**ID:** {root_node['id']}")
    st.write(f"**Тип:** {root_node['span_kind']}")
    st.write(f"**Количество дочерних нод:** {len(root_node['children'])}")

    st.subheader("📝 Разметка корневой ноды:")
    display_node_with_annotation(root_node, annotations)

    if st.button("💾 Сохранить все изменения для этой ноды", type="primary"):
        save_annotations(annotations)
        st.success("Все изменения сохранены!")


def main():
    st.title("🌳 Инструмент разметки нод")

    df = load_data()
    annotations = load_annotations()

    root_nodes, all_nodes = build_tree(df)

    if 'current_root_index' not in st.session_state:
        st.session_state.current_root_index = 0

    with st.sidebar:
        st.header("🧭 Навигация")

        st.subheader("📊 Статистика:")
        total_nodes = len(df)
        annotated_nodes = len([a for a in annotations.values() if a['approved'] is not None])
        approved_nodes = len([a for a in annotations.values() if a['approved'] is True])
        rejected_nodes = len([a for a in annotations.values() if a['approved'] is False])

        st.write(f"Всего нод: {total_nodes}")
        st.write(f"Размечено: {annotated_nodes}")
        st.write(f"Одобрено: {approved_nodes}")
        st.write(f"Отклонено: {rejected_nodes}")

        if total_nodes > 0:
            progress = annotated_nodes / total_nodes * 100
            st.progress(progress / 100)
            st.write(f"Прогресс: {progress:.1f}%")

        st.divider()

        st.subheader("🔍 Навигация:")
        current_index = st.session_state.current_root_index

        st.write(f"Корневая нода {current_index + 1} из {len(root_nodes)}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Предыдущая", disabled=current_index == 0):
                st.session_state.current_root_index = max(0, current_index - 1)
                st.rerun()

        with col2:
            if st.button("➡️ Следующая", disabled=current_index == len(root_nodes) - 1):
                st.session_state.current_root_index = min(len(root_nodes) - 1, current_index + 1)
                st.rerun()

        if len(root_nodes) > 0:
            jump_to = st.number_input(
                "Перейти к ноде №:",
                min_value=1,
                max_value=len(root_nodes),
                value=current_index + 1
            )
            if st.button("🚀 Перейти"):
                st.session_state.current_root_index = jump_to - 1
                st.rerun()

        st.divider()

        st.subheader("🔍 Фильтры:")

        span_kinds = ['Все'] + list(df['span_kind'].unique())
        span_filter = st.selectbox("Тип span:", span_kinds)

        search_term = st.text_input("🔍 Поиск по названию:")

        if span_filter != "Все" or search_term:
            filtered_indices = []
            for i, root in enumerate(root_nodes):
                if span_filter != "Все" and root['span_kind'] != span_filter:
                    continue
                if search_term and search_term.lower() not in root['name'].lower():
                    continue
                filtered_indices.append(i)

            if filtered_indices:
                st.write(f"Найдено {len(filtered_indices)} нод")
                if st.button("Показать первую найденную"):
                    st.session_state.current_root_index = filtered_indices[0]
                    st.rerun()
            else:
                st.warning("Нет нод, соответствующих фильтрам")

        st.divider()

        if st.button("💾 Сохранить все"):
            save_annotations(annotations)
            st.success("Сохранено!")

        if st.button("📤 Экспорт"):
            export_data = {
                'annotations': annotations,
                'export_time': datetime.now().isoformat(),
                'total_nodes': total_nodes
            }
            st.download_button(
                label="Скачать JSON",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"annotations_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    if len(root_nodes) == 0:
        st.warning("Нет корневых нод для отображения.")
        return

    current_index = st.session_state.current_root_index
    current_root = root_nodes[current_index]

    # Кнопки управления отображением
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📦 Свернуть все", key="collapse_all_page"):
            # Сворачиваем все ноды на странице
            for node_id in all_nodes:
                node_collapsed_key = f"node_collapsed_{node_id}"
                st.session_state[node_collapsed_key] = True
            st.rerun()
    
    with col2:
        if st.button("📂 Развернуть все", key="expand_all_page"):
            # Разворачиваем все ноды на странице
            for node_id in all_nodes:
                node_collapsed_key = f"node_collapsed_{node_id}"
                st.session_state[node_collapsed_key] = False
            st.rerun()
    
    with col3:
        if st.button("🔄 Сбросить состояние", key="reset_view"):
            # Сбрасываем состояние сворачивания к свернутому по умолчанию
            for key in list(st.session_state.keys()):
                if key.startswith('node_collapsed_'):
                    st.session_state[key] = True  # Все ноды свернуты
                elif key.startswith('children_expanded_'):
                    st.session_state[key] = False  # Все дочерние ноды свернуты
            st.rerun()
    
    with col4:
        if st.button("💾 Сохранить", key="save_all"):
            save_annotations(annotations)
            st.success("Сохранено!")

    display_root_node_with_children(current_root, all_nodes, annotations)

    st.subheader("⌨️ Быстрая навигация:")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Предыдущая нода", key="nav_prev"):
            st.session_state.current_root_index = max(0, current_index - 1)
            st.rerun()

    with col2:
        st.write(f"**{current_index + 1} / {len(root_nodes)}**")

    with col3:
        if st.button("➡️ Следующая нода", key="nav_next"):
            st.session_state.current_root_index = min(len(root_nodes) - 1, current_index + 1)
            st.rerun()


if __name__ == "__main__":
    main()
