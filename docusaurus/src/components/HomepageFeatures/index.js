import React from 'react';
import { features } from '../../data';
import { translate } from '@docusaurus/Translate';

export default function HomepageFeatures() {
  return (
    <div className="tw-relative tw-bg-white tw-py-8 sm:tw-py-12 lg:tw-py-16">
      <div className="tw-mx-auto tw-max-w-md tw-px-6 tw-text-center sm:tw-max-w-3xl lg:tw-max-w-7xl lg:tw-px-8">
        <p className="tw-mt-2 tw-text-3xl tw-font-bold tw-tracking-tight tw-text-gray-900 sm:tw-ext-4xl">
          {translate({ id: 'landing.features.title' })}
        </p>
        <div className="tw-mt-12">
          <div className="tw-grid tw-grid-cols-1 tw-gap-16 sm:tw-grid-cols-2 lg:tw-grid-cols-3">
            {features.map((feature) => (
              <div key={feature.name} className="pt-6">
                <div className="tw-flow-root tw-rounded-lg tw-px-6 tw-pb-8 tw-bg-gray-50 tw-drop-shadow-md">
                  <div className="tw--mt-6">
                    <div>
                      <span className="tw-inline-flex tw-items-center tw-justify-center tw-rounded-md tw-bg-gradient-to-r tw-from-purple-500 tw-to-pink-500 tw-p-3 tw-shadow-lg">
                        <feature.icon
                          className="tw-h-6 tw-w-6 tw-text-white"
                          aria-hidden="true"
                        />
                      </span>
                    </div>
                    <h3 className="tw-mt-8 tw-text-lg tw-font-medium tw-tracking-tight tw-text-gray-900">
                      {feature.name}
                    </h3>
                    <p className="tw-mt-5 tw-text-base tw-text-gray-500 tw-text-left">
                      {feature.description}
                    </p>
                    {feature.links && (
                      <ul className="tw-list-disc tw-text-left">
                        {feature.links.map((item, i) => {
                          return (
                            <li className="tw-text-left" key={i}>
                              <a
                                href={item.link}
                                className="tw-text-gray-500 hover:tw-no-underline">
                                {item.label}
                              </a>
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
