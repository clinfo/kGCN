package org.gcnk.knime.nodes.GCNScoreViewer;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "GCNScoreViewer" Node.
 * Show scores from output of GCNScore
 *
 * @author org.gcnk
 */
public class GCNScoreViewerNodeFactory 
        extends NodeFactory<GCNScoreViewerNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public GCNScoreViewerNodeModel createNodeModel() {
        return new GCNScoreViewerNodeModel();
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
    public NodeView<GCNScoreViewerNodeModel> createNodeView(final int viewIndex,
            final GCNScoreViewerNodeModel nodeModel) {
        return new GCNScoreViewerNodeView(nodeModel);
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
        return new GCNScoreViewerNodeDialog();
    }

}

