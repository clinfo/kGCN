package org.gcnk.knime.nodes.GCNVisualizer;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "GCNVisualizer" Node.
 * Generates data for displaying the contribution of each compound of the testset to the predicted value
 *
 * @author org.gcnk
 */
public class GCNVisualizerNodeFactory 
        extends NodeFactory<GCNVisualizerNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public GCNVisualizerNodeModel createNodeModel() {
        return new GCNVisualizerNodeModel();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getNrNodeViews() {
        return 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NodeView<GCNVisualizerNodeModel> createNodeView(final int viewIndex,
            final GCNVisualizerNodeModel nodeModel) {
        return new GCNVisualizerNodeView(nodeModel);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean hasDialog() {
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NodeDialogPane createNodeDialogPane() {
        return new GCNVisualizerNodeDialog();
    }

}

