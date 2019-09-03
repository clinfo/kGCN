package org.gcnk.knime.nodes.GCNDatasetBuilder;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "GCNDatasetBuilder" Node.
 * Read labels from CSV file and Create GCN Dataset together with graph structure and atom features.
 *
 * @author org.gcnk
 */
public class GCNDatasetBuilderNodeFactory 
        extends NodeFactory<GCNDatasetBuilderNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public GCNDatasetBuilderNodeModel createNodeModel() {
        return new GCNDatasetBuilderNodeModel();
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
    public NodeView<GCNDatasetBuilderNodeModel> createNodeView(final int viewIndex,
            final GCNDatasetBuilderNodeModel nodeModel) {
        return new GCNDatasetBuilderNodeView(nodeModel);
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
        return new GCNDatasetBuilderNodeDialog();
    }

}

