"""Fungsi untuk membuat visualisasi"""
import plotly.express as px
from constants import COLORS


def create_bar_chart(df, x, y, color, title, height=400):
    """Helper untuk membuat bar chart"""
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        barmode='group',
        color_discrete_map={'Tidak Stunting': COLORS['no_stunting'], 'Stunting': COLORS['stunting']},
        labels={'x': x, 'y': y}
    )
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True
    )
    return fig


def create_pie_chart(values, names, title, height=400):
    """Helper untuk membuat pie chart"""
    fig = px.pie(
        values=values,
        names=names,
        color=names,
        color_discrete_map={'Tidak Stunting': COLORS['no_stunting'], 'Stunting': COLORS['stunting']},
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(title=title, height=height, showlegend=True)
    return fig


def create_histogram(df, x, color_col, title, height=500):
    """Helper untuk membuat histogram"""
    fig = px.histogram(
        df,
        x=x,
        color=color_col,
        nbins=30,
        barmode='overlay',
        opacity=0.7,
        color_discrete_map={0: COLORS['no_stunting'], 1: COLORS['stunting']}
    )
    fig.update_layout(title=title, height=height)
    return fig


def create_scatter(df, x, y, color_col, size_col, title, height=600):
    """Helper untuk membuat scatter plot"""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color_col,
        size=size_col,
        hover_data=['Sex', 'ASI_Eksklusif'] if 'Sex' in df.columns else [],
        color_discrete_map={0: COLORS['no_stunting'], 1: COLORS['stunting']}
    )
    fig.update_layout(title=title, height=height)
    return fig


def create_box_plot(df, x, y, color_col, title, height=400):
    """Helper untuk membuat box plot"""
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color_col,
        color_discrete_map={0: COLORS['no_stunting'], 1: COLORS['stunting']}
    )
    fig.update_layout(title=title, height=height)
    return fig

