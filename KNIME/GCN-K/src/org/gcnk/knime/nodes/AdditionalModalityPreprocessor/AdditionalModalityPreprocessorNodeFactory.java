package org.gcnk.knime.nodes.AdditionalModalityPreprocessor;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "AdditionalModalityPreprocessor" Node.
 * Read new modality in CSV file format and generates output for AddModality node
 *
 * @author org.gcnk
 */
public class AdditionalModalityPreprocessorNodeFactory 
        extends NodeFactory<AdditionalModalityPreprocessorNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public AdditionalModalityPreprocessorNodeModel createNodeModel() {
        return new AdditionalModalityPreprocessorNodeModel();
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
    public NodeView<AdditionalModalityPreprocessorNodeModel> createNodeView(final int viewIndex,
            final AdditionalModalityPreprocessorNodeModel nodeModel) {
        return new AdditionalModalityPreprocessorNodeView(nodeModel);
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
        return new AdditionalModalityPreprocessorNodeDialog();
    }

}

