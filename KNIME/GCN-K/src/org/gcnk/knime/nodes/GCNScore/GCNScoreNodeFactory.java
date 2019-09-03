package org.gcnk.knime.nodes.GCNScore;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "GCNScore" Node.
 * Calculate scores from output of GCNPredictior
 *
 * @author org.gcnk
 */
public class GCNScoreNodeFactory 
        extends NodeFactory<GCNScoreNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public GCNScoreNodeModel createNodeModel() {
        return new GCNScoreNodeModel();
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
    public NodeView<GCNScoreNodeModel> createNodeView(final int viewIndex,
            final GCNScoreNodeModel nodeModel) {
        return new GCNScoreNodeView(nodeModel);
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
        return new GCNScoreNodeDialog();
    }

}

