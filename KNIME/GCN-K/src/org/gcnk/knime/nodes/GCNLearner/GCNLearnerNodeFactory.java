package org.gcnk.knime.nodes.GCNLearner;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "GCNLearner" Node.
 * Generate prediction model with training dataset.
 *
 * @author org.gcnk
 */
public class GCNLearnerNodeFactory 
        extends NodeFactory<GCNLearnerNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public GCNLearnerNodeModel createNodeModel() {
        return new GCNLearnerNodeModel();
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
    public NodeView<GCNLearnerNodeModel> createNodeView(final int viewIndex,
            final GCNLearnerNodeModel nodeModel) {
        return new GCNLearnerNodeView(nodeModel);
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
        return new GCNLearnerNodeDialog();
    }

}

