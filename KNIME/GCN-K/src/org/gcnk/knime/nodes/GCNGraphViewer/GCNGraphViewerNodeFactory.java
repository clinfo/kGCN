package org.gcnk.knime.nodes.GCNGraphViewer;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "GCNGraphViewer" Node.
 * Display graphically the contribution of each atom of each test compound in the test set to the predicted value on the compound from the output of the GCNVisualizer
 *
 * @author org.gcnk
 */
public class GCNGraphViewerNodeFactory 
        extends NodeFactory<GCNGraphViewerNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public GCNGraphViewerNodeModel createNodeModel() {
        return new GCNGraphViewerNodeModel();
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
    public NodeView<GCNGraphViewerNodeModel> createNodeView(final int viewIndex,
            final GCNGraphViewerNodeModel nodeModel) {
        return new GCNGraphViewerNodeView(nodeModel);
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
        return new GCNGraphViewerNodeDialog();
    }

}

