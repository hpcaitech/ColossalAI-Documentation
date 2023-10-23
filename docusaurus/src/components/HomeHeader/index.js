import React from 'react';
import { FaPlayCircle, FaGithub, FaSlack } from 'react-icons/fa';
import { translate } from '@docusaurus/Translate';

export default function HomeHeader() {
  return (
    <div className="tw-relative tw-isolate tw-bg-white">
      <div className="tw-absolute tw-inset-x-0 tw-top-[-10rem] tw--z-10 tw-transform-gputw-overflow-hidden tw-blur-3xl sm:tw-top-[-20rem]">
        <svg
          className="tw-relative tw-left-[calc(50%-11rem)] tw--z-10 tw-h-[21.1875rem] tw-max-w-none tw--translate-x-1/2 tw-rotate-[30deg] sm:tw-left-[calc(50%-30rem)] sm:tw-h-[42.375rem]"
          viewBox="0 0 1155 678">
          <path
            fill="url(#9b2541ea-d39d-499b-bd42-aeea3e93f5ff)"
            fillOpacity=".3"
            d="M317.219 518.975L203.852 678 0 438.341l317.219 80.634 204.172-286.402c1.307 132.337 45.083 346.658 209.733 145.248C936.936 126.058 882.053-94.234 1031.02 41.331c119.18 108.451 130.68 295.337 121.53 375.223L855 299l21.173 362.054-558.954-142.079z"
          />
          <defs>
            <linearGradient
              id="9b2541ea-d39d-499b-bd42-aeea3e93f5ff"
              x1="1155.49"
              x2="-78.208"
              y1=".177"
              y2="474.645"
              gradientUnits="userSpaceOnUse">
              <stop stopColor="#9089FC" />
              <stop offset={1} stopColor="#FF80B5" />
            </linearGradient>
          </defs>
        </svg>
      </div>
      <div className="tw-relative tw-px-6 lg:tw-px-8">
        <div className="tw-mx-auto tw-py-16 md:tw-py-24 lg:tw-py-36">
          <div className="tw-text-center">
            <h1 className="gradient-color tw-text-5xl tw-font-bold tw-tracking-tight tw-text-gray-900 sm:tw-text-4xl  md:tw-text-5xl lg:tw-text-7xl">
              {translate({ id: 'landing.header.title' })}
            </h1>
            <p className="tw-mt-6 tw-text-lg tw-leading-8 tw-text-gray-600">
              {translate({ id: 'landing.header.description' })}
            </p>
            <div className="tw-mt-10 tw-flex tw-flex-col md:tw-flex-row tw-items-center tw-justify-center tw-gap-x-6">
              <a
                href="docs/get_started/installation"
                className="tw-rounded-md tw-bg-indigo-600 tw-px-3.5 tw-py-1.5 tw-my-2 tw-text-base tw-font-semibold tw-leading-7 tw-text-white tw-shadow-sm hover:tw-bg-indigo-500 focus-visible:tw-outline focus-visible:tw-outline-2 focus-visible:tw-outline-offset-2 focus-visible:tw-outline-indigo-600 hover:tw-no-underline hover:tw-text-white">
                <div className="tw-flex tw-items-center">
                  <FaPlayCircle />
                  &nbsp;
                  {translate({ id: 'landing.header.buttons.get_started' })}
                </div>
              </a>
              <a
                href="https://github.com/hpcaitech/ColossalAI"
                className="tw-rounded-md tw-bg-indigo-600 tw-px-3.5 tw-py-1.5 tw-my-2 tw-text-base tw-font-semibold tw-leading-7 tw-text-white tw-shadow-sm hover:tw-bg-indigo-500 focus-visible:tw-outline focus-visible:tw-outline-2 focus-visible:tw-outline-offset-2 focus-visible:tw-outline-indigo-600 hover:tw-no-underline hover:tw-text-white">
                <div className="tw-flex tw-items-center">
                  <FaGithub />
                  &nbsp;{translate({ id: 'landing.header.buttons.github' })}
                </div>
              </a>
              <a
                href="https://github.com/hpcaitech/public_assets/tree/main/colossalai/contact/slack"
                className="tw-rounded-md tw-bg-indigo-600 tw-px-3.5 tw-py-1.5 tw-my-2 tw-text-base tw-font-semibold tw-leading-7 tw-text-white tw-shadow-sm hover:tw-bg-indigo-500 focus-visible:tw-outline focus-visible:tw-outline-2 focus-visible:tw-outline-offset-2 focus-visible:tw-outline-indigo-600 hover:tw-no-underline hover:tw-text-white">
                <div className="tw-flex tw-items-center">
                  <FaSlack />
                  &nbsp;{translate({ id: 'landing.header.buttons.slack' })}
                </div>
              </a>
            </div>
          </div>
          <div className="tw-hidden tw-mt-8 sm:tw-mb-8 sm:tw-flex sm:tw-justify-center">
            <div className="tw-relative tw-rounded-full tw-py-1 tw-px-3 tw-text-sm tw-leading-6 tw-text-gray-600 tw-ring-1 tw-ring-gray-900/10 hover:tw-ring-gray-900/20">
              {translate({ id: 'landing.header.question' })}{' '}
              <a
                href="https://www.hpc-ai.tech/contact"
                className="tw-font-semibold tw-text-indigo-600">
                <span className="tw-absolute tw-inset-0" aria-hidden="true" />
                {translate({ id: 'landing.header.talk_to_expert' })}{' '}
                <span aria-hidden="true">&rarr;</span>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
