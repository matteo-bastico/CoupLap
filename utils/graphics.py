import plotly.graph_objects as go


MARKER = dict(
    size=1,
    opacity=0.7
)
MARKER_LARGE = dict(
    size=10,
    opacity=0.7
)


def plot_pointcloud(
    points,
    fig=None,
    name=None,
    marker=None,
    row=None,
    col=None,
    axis=True,
    camera=None,
    bgcolor='rgba(0,0,0,0)'
):
    """
    Function to generate a 3D scatter of a pointcloud in Plotly.

    :param points: Nx3 np.Array of 3D points
    :param fig: default=None, if not None the new pointcloud is added to the given figure
    :param name: default=None, pointcloud name
    :param marker: default=None, dict of marker description (check plotly docs)
    :param row: default=None, row number if the given fig contains subplots
    :param col: default=None, column number if the given fig contains subplots
    :param axis: default=True, if True the axis are visible
    :param camera: default=None, dict for camera orientation (check plotly docs)
    :param bgcolor: default='rgba(0,0,0,0)', background color
    :return:
    """
    data = [go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        name=name,
        marker=marker
    )]

    if fig is None:
        fig = go.Figure(
            data=data,
        )
    else:
        fig.add_trace(data[0], row=row, col=col)

    fig.update_scenes(
        xaxis=dict(visible=axis),
        yaxis=dict(visible=axis),
        zaxis=dict(visible=axis),
        bgcolor=bgcolor,
        camera=camera,
        aspectmode="data"
    )
    return fig


