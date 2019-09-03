package org.gcnk.knime.nodes.GCNPredictor;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "GCNPredictor" Node.
 * Predict test dataset with model.
 *
 * @author org.gcnk
 */
public class GCNPredictorNodeFactory 
        extends NodeFactory<GCNPredictorNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public GCNPredictorNodeModel createNodeModel() {
        return new GCNPredictorNodeModel();
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
    public NodeView<GCNPredictorNodeModel> createNodeView(final int viewIndex,
            final GCNPredictorNodeModel nodeModel) {
        return new GCNPredictorNodeView(nodeModel);
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
        return new GCNPredictorNodeDialog();
    }

}

