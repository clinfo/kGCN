package org.gcnk.knime.nodes.GCNDatasetSplitter;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "GCNDatasetSplitter" Node.
 * Split dataset into two parts.
 *
 * @author org.gcnk
 */
public class GCNDatasetSplitterNodeFactory 
        extends NodeFactory<GCNDatasetSplitterNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public GCNDatasetSplitterNodeModel createNodeModel() {
        return new GCNDatasetSplitterNodeModel();
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
    public NodeView<GCNDatasetSplitterNodeModel> createNodeView(final int viewIndex,
            final GCNDatasetSplitterNodeModel nodeModel) {
        return new GCNDatasetSplitterNodeView(nodeModel);
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
        return new GCNDatasetSplitterNodeDialog();
    }

}

